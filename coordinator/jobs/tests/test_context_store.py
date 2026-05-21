"""Tests for the context store and context merger modules.

Tests cover:
- ContextStore CRUD operations (with mock Redis)
- Context window building (5-page windows)
- Reference ID format and uniqueness
- TTL configuration
- Payload size validation
- Backward compatibility (tasks without context)
- ContextMerger paragraph continuation detection
- ContextMerger table continuation detection
- Cross-page paragraph merging
- Cross-page table row merging

Run with: cd coordinator && python -m pytest jobs/tests/test_context_store.py -v
"""

import json
import time
import uuid
from unittest.mock import patch

import pytest

from jobs.context_merger import (
    ContextMerger,
    _count_consistent_spaces,
    _extract_first_paragraph,
    _extract_last_paragraph,
    detect_continued_paragraph,
    detect_continued_table,
    merge_paragraphs,
    merge_table_rows,
)
from jobs.context_store import (
    ContextStore,
    _get_config,
    _job_key_pattern,
    _make_ref_id,
    _redis_key,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class FakeRedis:
    """In-memory fake Redis for unit testing without a real Redis server."""

    def __init__(self):
        self._store: dict[str, bytes] = {}
        self._expiry: dict[str, float] = {}
        self._sets: dict[str, set[bytes]] = {}

    def set(self, key, value, ex=None):
        if isinstance(value, str):
            value = value.encode("utf-8")
        self._store[key] = value
        if ex is not None:
            self._expiry[key] = time.time() + ex
        return True

    def get(self, key):
        if key in self._expiry and time.time() > self._expiry[key]:
            del self._store[key]
            del self._expiry[key]
            return None
        return self._store.get(key)

    def delete(self, *keys):
        count = 0
        for key in keys:
            if key in self._store:
                del self._store[key]
                count += 1
            if key in self._sets:
                del self._sets[key]
                count += 1
            if key in self._expiry:
                del self._expiry[key]
        return count

    def sadd(self, key, *values):
        if key not in self._sets:
            self._sets[key] = set()
        added = 0
        for v in values:
            if isinstance(v, str):
                v = v.encode("utf-8")
            if v not in self._sets[key]:
                self._sets[key].add(v)
                added += 1
        return added

    def smembers(self, key):
        return self._sets.get(key, set())

    def expire(self, key, seconds):
        if key in self._store or key in self._sets:
            self._expiry[key] = time.time() + seconds
            return True
        return False


@pytest.fixture
def fake_redis():
    return FakeRedis()


@pytest.fixture
def context_store(fake_redis):
    """ContextStore with fake Redis and explicit config."""
    store = ContextStore(
        redis_url="redis://localhost:6379/0",
        ttl_seconds=3600,
        window_size=5,
        _redis_client=fake_redis,
    )
    store._enabled = True
    return store


@pytest.fixture
def sample_pages():
    """Create a 7-page sample document for context window testing."""
    pages = []
    for i in range(1, 8):
        pages.append({
            "page_num": i,
            "text": f"This is the full text of page {i}. "
                    f"It contains important content about topic {i}.",
            "ocr_confidence": 0.95,
            "layout_regions": [
                {"type": "text", "bbox": [10, 10, 500, 700]},
            ],
        })
    return pages


@pytest.fixture
def merger():
    return ContextMerger()


# ===========================================================================
# Reference ID tests
# ===========================================================================

class TestRefId:
    def test_ref_id_format(self):
        ref = _make_ref_id("550e8400-e29b-41d4-a716-446655440000", 5)
        parts = ref.split(":")
        assert len(parts) == 4
        assert parts[0] == "ctx"
        assert parts[2] == "5"
        assert len(parts[3]) == 8  # short UUID

    def test_ref_id_uniqueness(self):
        refs = {_make_ref_id("abc-123", 1) for _ in range(100)}
        assert len(refs) == 100, "Reference IDs should be unique"

    def test_ref_id_different_pages(self):
        ref_a = _make_ref_id("abc-123", 1)
        ref_b = _make_ref_id("abc-123", 2)
        assert ref_a != ref_b

    def test_redis_key_format(self):
        ref = "ctx:abc123:1:deadbeef"
        key = _redis_key(ref)
        assert key == "context:ctx:abc123:1:deadbeef"

    def test_job_key_pattern(self):
        pattern = _job_key_pattern("550e8400-e29b-41d4-a716-446655440000")
        assert pattern.startswith("ctx:")
        assert pattern.endswith(":*")


# ===========================================================================
# Configuration tests
# ===========================================================================

def _patch_django_settings_none():
    """Patch Django settings attrs to None so env var fallback is exercised."""
    from django.conf import settings as django_settings
    patches = []
    for attr in ("CONTEXT_WINDOW_ENABLED", "CONTEXT_WINDOW_SIZE",
                 "CONTEXT_STORE_TTL", "CONTEXT_STORE_URL", "REDIS_URL"):
        if hasattr(django_settings, attr):
            patches.append(patch.object(django_settings, attr, None))
    return patches


class TestConfig:
    def test_default_config(self):
        with patch.dict("os.environ", {}, clear=False):
            config = _get_config()
            assert isinstance(config["enabled"], bool)
            assert config["window_size"] >= 1
            assert config["ttl"] >= 60
            assert isinstance(config["redis_url"], str)

    def test_django_settings_config(self):
        """When Django settings are set, they take precedence over env vars."""
        config = _get_config()
        # settings_test.py sets CONTEXT_WINDOW_ENABLED = False
        assert config["enabled"] is False
        assert config["window_size"] == 5
        assert config["ttl"] == 3600

    def test_env_override_enabled(self):
        patches = _patch_django_settings_none()
        for p in patches:
            p.start()
        try:
            with patch.dict("os.environ", {"CONTEXT_WINDOW_ENABLED": "true"}, clear=False):
                config = _get_config()
                assert config["enabled"] is True
        finally:
            for p in patches:
                p.stop()

    def test_env_override_window_size(self):
        patches = _patch_django_settings_none()
        for p in patches:
            p.start()
        try:
            with patch.dict("os.environ", {"CONTEXT_WINDOW_SIZE": "7"}, clear=False):
                config = _get_config()
                assert config["window_size"] == 7
        finally:
            for p in patches:
                p.stop()

    def test_env_override_ttl(self):
        patches = _patch_django_settings_none()
        for p in patches:
            p.start()
        try:
            with patch.dict("os.environ", {"CONTEXT_STORE_TTL": "7200"}, clear=False):
                config = _get_config()
                assert config["ttl"] == 7200
        finally:
            for p in patches:
                p.stop()

    def test_env_override_url(self):
        patches = _patch_django_settings_none()
        for p in patches:
            p.start()
        try:
            with patch.dict("os.environ", {"CONTEXT_STORE_URL": "redis://custom:6380/1"}, clear=False):
                config = _get_config()
                assert config["redis_url"] == "redis://custom:6380/1"
        finally:
            for p in patches:
                p.stop()

    def test_invalid_window_size_uses_default(self):
        patches = _patch_django_settings_none()
        for p in patches:
            p.start()
        try:
            with patch.dict("os.environ", {"CONTEXT_WINDOW_SIZE": "bad"}, clear=False):
                config = _get_config()
                assert config["window_size"] == 5  # default
        finally:
            for p in patches:
                p.stop()

    def test_minimum_ttl_clamped(self):
        patches = _patch_django_settings_none()
        for p in patches:
            p.start()
        try:
            with patch.dict("os.environ", {"CONTEXT_STORE_TTL": "10"}, clear=False):
                config = _get_config()
                assert config["ttl"] >= 60
        finally:
            for p in patches:
                p.stop()


# ===========================================================================
# ContextStore CRUD tests
# ===========================================================================

class TestContextStoreCRUD:
    def test_store_and_retrieve(self, context_store, fake_redis):
        context = {"text": "Hello world", "page_num": 1}
        ref_id = context_store.store_page_context("job-1", 1, context)

        assert ref_id.startswith("ctx:")
        result = context_store.get_page_context(ref_id)
        assert result is not None
        assert result["text"] == "Hello world"
        assert result["page_num"] == 1

    def test_retrieve_missing_returns_none(self, context_store):
        result = context_store.get_page_context("ctx:nonexistent:1:abcdef12")
        assert result is None

    def test_delete_context(self, context_store):
        ref_id = context_store.store_page_context("job-1", 1, {"text": "test"})
        assert context_store.get_page_context(ref_id) is not None

        deleted = context_store.delete_context(ref_id)
        assert deleted is True
        assert context_store.get_page_context(ref_id) is None

    def test_delete_nonexistent_returns_false(self, context_store):
        deleted = context_store.delete_context("ctx:nonexistent:1:abcdef12")
        assert deleted is False

    def test_cleanup_job(self, context_store):
        job_id = "550e8400-e29b-41d4-a716-446655440000"
        refs = []
        for page in range(1, 6):
            ref = context_store.store_page_context(
                job_id, page, {"text": f"page {page}"}
            )
            refs.append(ref)

        # All should be retrievable
        for ref in refs:
            assert context_store.get_page_context(ref) is not None

        # Cleanup
        deleted = context_store.cleanup_job(job_id)
        assert deleted >= 5  # at least 5 context keys + 1 set key

        # All should be gone
        for ref in refs:
            assert context_store.get_page_context(ref) is None

    def test_cleanup_nonexistent_job(self, context_store):
        deleted = context_store.cleanup_job("nonexistent-job-id")
        assert deleted == 0

    def test_store_complex_context(self, context_store):
        """Context can contain nested dicts, lists, and various types."""
        context = {
            "text": "Complex page",
            "page_num": 3,
            "ocr_lines": [
                ("line 1", [10, 20, 100, 30], 0.95),
                ("line 2", [10, 40, 100, 50], 0.88),
            ],
            "layout_regions": [
                {"type": "table", "bbox": [0, 0, 500, 200], "confidence": 0.92},
            ],
            "confidence": 0.91,
        }
        ref_id = context_store.store_page_context("job-complex", 3, context)
        result = context_store.get_page_context(ref_id)
        assert result["page_num"] == 3
        assert len(result["ocr_lines"]) == 2
        assert result["layout_regions"][0]["type"] == "table"


# ===========================================================================
# Context window building tests
# ===========================================================================

class TestContextWindowBuilding:
    def test_build_middle_page_window(self, context_store, sample_pages):
        """Page 4 of 7: should have previous, current, and next."""
        result = context_store.build_context_window("job-1", sample_pages, 4)

        assert result["target_page_num"] == 4
        assert result["total_pages"] == 7
        assert result["current"]["page_num"] == 4
        assert result["previous"]["page_num"] == 3
        assert result["next"]["page_num"] == 5
        assert "ref_id" in result

    def test_build_first_page_window(self, context_store, sample_pages):
        """Page 1: no previous page."""
        result = context_store.build_context_window("job-1", sample_pages, 1)

        assert result["target_page_num"] == 1
        assert result["current"]["page_num"] == 1
        assert result["previous"] is None
        assert result["next"]["page_num"] == 2

    def test_build_last_page_window(self, context_store, sample_pages):
        """Page 7: no next page."""
        result = context_store.build_context_window("job-1", sample_pages, 7)

        assert result["target_page_num"] == 7
        assert result["current"]["page_num"] == 7
        assert result["previous"]["page_num"] == 6
        assert result["next"] is None

    def test_build_single_page_document(self, context_store):
        """Single-page doc: no previous, no next."""
        pages = [{"page_num": 1, "text": "Only page"}]
        result = context_store.build_context_window("job-1", pages, 1)

        assert result["previous"] is None
        assert result["next"] is None
        assert result["current"]["text"] == "Only page"

    def test_build_window_has_summaries(self, context_store, sample_pages):
        """Middle page should have both summary_before and summary_after."""
        result = context_store.build_context_window("job-1", sample_pages, 4)

        # summary_before covers pages before the previous page
        assert result["summary_before"] is not None
        assert result["summary_before"]["page_count"] >= 1

        # summary_after covers pages after the next page
        assert result["summary_after"] is not None
        assert result["summary_after"]["page_count"] >= 1

    def test_build_window_invalid_page_raises(self, context_store, sample_pages):
        with pytest.raises(ValueError, match="out of range"):
            context_store.build_context_window("job-1", sample_pages, 0)

        with pytest.raises(ValueError, match="out of range"):
            context_store.build_context_window("job-1", sample_pages, 99)

    def test_build_all_context_windows(self, context_store, sample_pages):
        ref_ids = context_store.build_all_context_windows("job-1", sample_pages)

        assert len(ref_ids) == 7
        for ref in ref_ids:
            assert ref.startswith("ctx:")
            # Each should be retrievable
            ctx = context_store.get_page_context(ref)
            assert ctx is not None

    def test_context_is_stored_in_redis(self, context_store, sample_pages, fake_redis):
        """Verify the context is actually persisted in Redis."""
        result = context_store.build_context_window("job-1", sample_pages, 3)
        ref_id = result["ref_id"]

        # Should be in fake_redis store
        key = f"context:{ref_id}"
        raw = fake_redis.get(key)
        assert raw is not None
        data = json.loads(raw)
        assert data["target_page_num"] == 3

    def test_window_has_timestamp(self, context_store, sample_pages):
        result = context_store.build_context_window("job-1", sample_pages, 1)
        assert "timestamp" in result
        assert isinstance(result["timestamp"], float)


# ===========================================================================
# Summary building tests
# ===========================================================================

class TestSummaryBuilding:
    def test_summary_with_pages(self):
        pages = [
            {"page_num": 1, "text": "First page content here."},
            {"page_num": 2, "text": "Second page content here."},
            {"page_num": 3, "text": "Third page content here."},
        ]
        summary = ContextStore._build_summary(pages, 0, 3)
        assert summary is not None
        assert summary["page_count"] == 3
        assert summary["page_range"] == [1, 2, 3]
        assert summary["total_text_length"] > 0
        assert len(summary["first_lines"]) <= 5

    def test_summary_empty_range(self):
        pages = [{"page_num": 1, "text": "test"}]
        summary = ContextStore._build_summary(pages, 1, 1)
        assert summary is None

    def test_summary_caps_first_lines(self):
        pages = [{"page_num": i, "text": f"Line {i}"} for i in range(1, 20)]
        summary = ContextStore._build_summary(pages, 0, 20)
        assert summary is not None
        assert len(summary["first_lines"]) <= 5


# ===========================================================================
# Payload size validation tests
# ===========================================================================

class TestPayloadValidation:
    def test_small_payload_passes(self):
        payload = {"job_id": "abc", "page_num": 1, "context_ref_id": "ctx:abc:1:def"}
        assert ContextStore.validate_broker_payload(payload) is True

    def test_large_payload_fails(self):
        payload = {"data": "x" * 20000}
        assert ContextStore.validate_broker_payload(payload, max_bytes=10240) is False

    def test_ref_id_payload_under_10kb(self):
        """A typical Celery message with ref_id instead of data stays under 10KB."""
        payload = {
            "job_id": str(uuid.uuid4()),
            "page_num": 42,
            "context_ref_id": "ctx:550e8400e29b:42:abcdef12",
            "presigned_urls": {
                "source_get": "https://s3.example.com/signed-url-here?token=abc123",
                "page_pdf_put": "https://s3.example.com/signed-url-here?token=def456",
                "page_text_put": "https://s3.example.com/signed-url-here?token=ghi789",
            },
        }
        size = ContextStore.measure_payload_size(payload)
        assert size < 10240, f"Ref-based payload is {size} bytes, should be < 10KB"

    def test_measure_payload_size(self):
        payload = {"key": "value"}
        size = ContextStore.measure_payload_size(payload)
        assert size == len(json.dumps(payload).encode("utf-8"))


# ===========================================================================
# Backward compatibility tests
# ===========================================================================

class TestBackwardCompatibility:
    def test_tasks_work_without_context_ref(self):
        """Existing task payloads without context_ref_id should be valid."""
        old_style_payload = {
            "job_id": str(uuid.uuid4()),
            "page_num": 1,
        }
        # Should validate fine
        assert ContextStore.validate_broker_payload(old_style_payload) is True

    def test_disabled_store_returns_none_on_get(self, fake_redis):
        """When disabled, the store can still be called safely."""
        store = ContextStore(
            ttl_seconds=3600,
            window_size=5,
            _redis_client=fake_redis,
        )
        store._enabled = False
        # Store and retrieve still work -- the enabled flag is advisory
        ref_id = store.store_page_context("job-1", 1, {"text": "test"})
        result = store.get_page_context(ref_id)
        assert result is not None


# ===========================================================================
# Paragraph continuation detection tests
# ===========================================================================

class TestParagraphContinuation:
    def test_sentence_ends_with_comma(self):
        assert detect_continued_paragraph(
            "This sentence continues,",
            "and finishes here.",
        ) is True

    def test_sentence_ends_with_semicolon(self):
        assert detect_continued_paragraph(
            "First clause;",
            "second clause follows.",
        ) is True

    def test_sentence_ends_with_hyphen(self):
        assert detect_continued_paragraph(
            "The document con-",
            "tinues on the next page.",
        ) is True

    def test_next_page_starts_lowercase(self):
        assert detect_continued_paragraph(
            "The paragraph ends without period",
            "and the next page continues.",
        ) is True

    def test_proper_sentence_end(self):
        """A sentence that ends properly should not trigger continuation."""
        assert detect_continued_paragraph(
            "This sentence ends properly.",
            "A new sentence begins here.",
        ) is False

    def test_exclamation_end(self):
        assert detect_continued_paragraph(
            "This ends with excitement!",
            "New content starts.",
        ) is False

    def test_question_end(self):
        assert detect_continued_paragraph(
            "Does this end with a question?",
            "Yes it does.",
        ) is False

    def test_empty_inputs(self):
        assert detect_continued_paragraph("", "Next page") is False
        assert detect_continued_paragraph("Current page", "") is False
        assert detect_continued_paragraph("", "") is False

    def test_short_last_line_without_terminal(self):
        """Short last line without terminal punctuation suggests mid-sentence."""
        assert detect_continued_paragraph(
            "Previous content.\nThe data shows that\nvalues are",
            "significantly higher than expected.",
        ) is True


# ===========================================================================
# Table continuation detection tests
# ===========================================================================

class TestTableContinuation:
    def test_pipe_delimited_table(self):
        end_text = "| Name | Age | City |\n| John | 30 | NYC |"
        start_text = "| Jane | 25 | LA |\n| Bob | 35 | SF |"
        assert detect_continued_table(end_text, start_text) is True

    def test_tab_separated_table(self):
        end_text = "Name\tAge\tCity\nJohn\t30\tNYC"
        start_text = "Jane\t25\tLA\nBob\t35\tSF"
        assert detect_continued_table(end_text, start_text) is True

    def test_no_table_markers(self):
        end_text = "This is regular text at the end."
        start_text = "This is regular text at the start."
        assert detect_continued_table(end_text, start_text) is False

    def test_empty_inputs(self):
        assert detect_continued_table("", "table|data") is False
        assert detect_continued_table("table|data", "") is False

    def test_mixed_table_and_text(self):
        """Only table on one side should not trigger continuation."""
        end_text = "This is just plain text without any table markers."
        start_text = "| Column1 | Column2 |"
        assert detect_continued_table(end_text, start_text) is False


# ===========================================================================
# Paragraph merging tests
# ===========================================================================

class TestParagraphMerging:
    def test_hyphenated_merge(self):
        result = merge_paragraphs("The docu-", "ment continues.")
        assert result == "The document continues."

    def test_simple_join(self):
        result = merge_paragraphs("End of page", "start of next page")
        assert result == "End of page start of next page"

    def test_empty_current(self):
        result = merge_paragraphs("", "next page text")
        assert result == "next page text"

    def test_empty_next(self):
        result = merge_paragraphs("current page text", "")
        assert result == "current page text"

    def test_preserves_content(self):
        result = merge_paragraphs(
            "The quick brown fox jumps over",
            "the lazy dog.",
        )
        assert "quick brown fox" in result
        assert "lazy dog" in result


# ===========================================================================
# Table row merging tests
# ===========================================================================

class TestTableRowMerging:
    def test_simple_merge(self):
        current = [["Name", "Age"], ["John", "30"]]
        next_rows = [["Jane", "25"], ["Bob", "35"]]
        result = merge_table_rows(current, next_rows)
        assert len(result) == 4

    def test_skip_repeated_header(self):
        current = [["Name", "Age"], ["John", "30"]]
        next_rows = [["Name", "Age"], ["Jane", "25"]]
        result = merge_table_rows(current, next_rows, skip_repeated_header=True)
        assert len(result) == 3  # header skipped
        assert result[0] == ["Name", "Age"]
        assert result[1] == ["John", "30"]
        assert result[2] == ["Jane", "25"]

    def test_no_skip_when_disabled(self):
        current = [["Name", "Age"], ["John", "30"]]
        next_rows = [["Name", "Age"], ["Jane", "25"]]
        result = merge_table_rows(current, next_rows, skip_repeated_header=False)
        assert len(result) == 4  # header kept

    def test_skip_header_plus_separator(self):
        current = [["Name", "Age"], ["---", "---"], ["John", "30"]]
        next_rows = [["Name", "Age"], ["---", "---"], ["Jane", "25"]]
        result = merge_table_rows(current, next_rows, skip_repeated_header=True)
        # Should skip repeated header and separator
        assert len(result) == 4  # 3 from current + 1 data row from next

    def test_empty_current(self):
        result = merge_table_rows([], [["a", "b"]])
        assert result == [["a", "b"]]

    def test_empty_next(self):
        result = merge_table_rows([["a", "b"]], [])
        assert result == [["a", "b"]]

    def test_different_headers_not_skipped(self):
        current = [["Name", "Age"], ["John", "30"]]
        next_rows = [["City", "State"], ["NYC", "NY"]]
        result = merge_table_rows(current, next_rows, skip_repeated_header=True)
        assert len(result) == 4  # headers differ, so nothing skipped


# ===========================================================================
# ContextMerger tests
# ===========================================================================

class TestContextMerger:
    def test_merge_with_continuation(self, merger):
        context = {
            "target_page_num": 2,
            "current": {
                "text": "and the analysis shows that results\n\nSecond paragraph.",
            },
            "previous": {
                "text": "The document discusses important findings,",
            },
            "next": {
                "text": "Furthermore, the data indicates\n\nNew section.",
            },
        }
        result = merger.merge_from_context(context)

        assert result["page_num"] == 2
        assert result["paragraph_continuation_from_previous"] is True
        assert result["merged_text_prefix"] is not None

    def test_merge_no_continuation(self, merger):
        context = {
            "target_page_num": 2,
            "current": {
                "text": "Complete sentence on this page.",
            },
            "previous": {
                "text": "Previous page ends cleanly.",
            },
            "next": {
                "text": "Next page starts a new topic.",
            },
        }
        result = merger.merge_from_context(context)

        assert result["paragraph_continuation_from_previous"] is False
        assert result["paragraph_continuation_to_next"] is False

    def test_merge_first_page(self, merger):
        """First page has no previous."""
        context = {
            "target_page_num": 1,
            "current": {"text": "First page content."},
            "previous": None,
            "next": {"text": "Next page starts fresh."},
        }
        result = merger.merge_from_context(context)
        assert result["paragraph_continuation_from_previous"] is False

    def test_merge_last_page(self, merger):
        """Last page has no next."""
        context = {
            "target_page_num": 5,
            "current": {"text": "Last page content."},
            "previous": {"text": "Previous page."},
            "next": None,
        }
        result = merger.merge_from_context(context)
        assert result["paragraph_continuation_to_next"] is False

    def test_merge_table_continuation(self, merger):
        context = {
            "target_page_num": 2,
            "current": {
                "text": "| Col1 | Col2 |\n| data1 | data2 |",
                "table_rows": [["Col1", "Col2"], ["data1", "data2"]],
            },
            "previous": {
                "text": "| Col1 | Col2 |\n| prev1 | prev2 |",
                "table_rows": [["Col1", "Col2"], ["prev1", "prev2"]],
            },
            "next": None,
        }
        result = merger.merge_from_context(context)
        assert result["table_continuation_from_previous"] is True
        assert result["merged_table_rows"] is not None
        # Should have merged rows with header dedup
        assert len(result["merged_table_rows"]) == 3

    def test_merge_empty_context(self, merger):
        context = {
            "target_page_num": 1,
            "current": {},
            "previous": None,
            "next": None,
        }
        result = merger.merge_from_context(context)
        assert result["page_num"] == 1
        assert result["paragraph_continuation_from_previous"] is False


# ===========================================================================
# Helper function tests
# ===========================================================================

class TestHelpers:
    def test_extract_last_paragraph(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nLast paragraph here."
        result = _extract_last_paragraph(text)
        assert result == "Last paragraph here."

    def test_extract_first_paragraph(self):
        text = "First paragraph here.\n\nSecond paragraph."
        result = _extract_first_paragraph(text)
        assert result == "First paragraph here."

    def test_extract_last_paragraph_single(self):
        assert _extract_last_paragraph("Only one paragraph.") == "Only one paragraph."

    def test_extract_first_paragraph_single(self):
        assert _extract_first_paragraph("Only one paragraph.") == "Only one paragraph."

    def test_extract_from_empty(self):
        assert _extract_last_paragraph("") == ""
        assert _extract_first_paragraph("") == ""

    def test_count_consistent_spaces_aligned(self):
        lines = [
            "Name    Age   City",
            "John    30    NYC",
            "Jane    25    LA",
        ]
        count = _count_consistent_spaces(lines)
        assert count >= 2  # At least 2 aligned column positions

    def test_count_consistent_spaces_unaligned(self):
        lines = ["This is a normal sentence."]
        count = _count_consistent_spaces(lines)
        assert count == 0  # Single line, no alignment


# ===========================================================================
# Integration-style tests
# ===========================================================================

class TestEndToEnd:
    def test_full_workflow(self, context_store, sample_pages):
        """Full workflow: build windows, retrieve, merge, cleanup."""
        job_id = str(uuid.uuid4())

        # Build all context windows
        ref_ids = context_store.build_all_context_windows(job_id, sample_pages)
        assert len(ref_ids) == 7

        # Retrieve a middle page context
        ctx = context_store.get_page_context(ref_ids[3])  # page 4
        assert ctx is not None
        assert ctx["target_page_num"] == 4
        assert ctx["previous"] is not None
        assert ctx["next"] is not None

        # Run merger
        merger = ContextMerger()
        merge_result = merger.merge_from_context(ctx)
        assert merge_result["page_num"] == 4

        # Cleanup
        deleted = context_store.cleanup_job(job_id)
        assert deleted >= 7

        # Verify cleanup
        for ref in ref_ids:
            assert context_store.get_page_context(ref) is None

    def test_ref_id_in_task_payload(self, context_store, sample_pages):
        """Simulate a Celery task payload with context reference ID."""
        job_id = str(uuid.uuid4())
        ref_ids = context_store.build_all_context_windows(job_id, sample_pages)

        # Build a task payload like extract_pages would
        task_payload = {
            "job_id": job_id,
            "page_num": 3,
            "context_ref_id": ref_ids[2],
            "presigned_urls": None,
        }

        # Validate size
        assert ContextStore.validate_broker_payload(task_payload) is True
        size = ContextStore.measure_payload_size(task_payload)
        assert size < 10240

        # Worker retrieves context
        ctx = context_store.get_page_context(task_payload["context_ref_id"])
        assert ctx is not None
        assert ctx["target_page_num"] == 3

    def test_two_page_document(self, context_store):
        """Edge case: 2-page document."""
        pages = [
            {"page_num": 1, "text": "Page one text."},
            {"page_num": 2, "text": "Page two text."},
        ]
        ref_ids = context_store.build_all_context_windows("job-2pg", pages)
        assert len(ref_ids) == 2

        ctx1 = context_store.get_page_context(ref_ids[0])
        assert ctx1["previous"] is None
        assert ctx1["next"]["page_num"] == 2

        ctx2 = context_store.get_page_context(ref_ids[1])
        assert ctx2["previous"]["page_num"] == 1
        assert ctx2["next"] is None
