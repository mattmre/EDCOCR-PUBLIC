"""Tests for the review queue data model and storage layer."""

from __future__ import annotations

import sqlite3
from unittest.mock import patch

import pytest

from api.review_queue import ReviewItem, ReviewQueue, ReviewReason, ReviewStatus


@pytest.fixture()
def queue(tmp_path):
    """Provide a ReviewQueue backed by a temporary database."""
    db = str(tmp_path / "review_test.db")
    with patch("api.review_queue.DB_PATH", db):
        q = ReviewQueue(db_path=db)
        yield q


# ---------------------------------------------------------------------------
# ReviewItem dataclass
# ---------------------------------------------------------------------------


class TestReviewItem:
    def test_to_dict_round_trip(self):
        item = ReviewItem(
            review_id="rev_aabbccddee01",
            job_id="job_aabbccddee01",
            reason="low_confidence",
            confidence=0.42,
            quality_classification="degraded",
            status="pending",
            reviewer="",
            decision_notes="",
            created_at="2026-03-29T00:00:00.000+00:00",
            reviewed_at="",
            metadata={"pages": 5},
        )
        d = item.to_dict()
        assert d["review_id"] == "rev_aabbccddee01"
        assert d["confidence"] == 0.42
        assert d["metadata"] == {"pages": 5}
        assert d["status"] == "pending"

    def test_from_row(self):
        """Simulate sqlite3.Row with a dict-like wrapper."""
        row_data = {
            "review_id": "rev_112233445566",
            "job_id": "job_112233445566",
            "reason": "degraded_quality",
            "confidence": 0.35,
            "quality_classification": "review_required",
            "status": "approved",
            "reviewer": "alice",
            "decision_notes": "Looks OK upon inspection",
            "created_at": "2026-03-29T00:00:00.000+00:00",
            "reviewed_at": "2026-03-29T01:00:00.000+00:00",
            "metadata_json": '{"source": "test"}',
        }

        # Use an actual SQLite row
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            "CREATE TABLE t ("
            "review_id TEXT, job_id TEXT, reason TEXT, confidence REAL, "
            "quality_classification TEXT, status TEXT, reviewer TEXT, "
            "decision_notes TEXT, created_at TEXT, reviewed_at TEXT, "
            "metadata_json TEXT)"
        )
        conn.execute(
            "INSERT INTO t VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            tuple(row_data.values()),
        )
        row = conn.execute("SELECT * FROM t").fetchone()

        item = ReviewItem.from_row(row)
        assert item.review_id == "rev_112233445566"
        assert item.reason == "degraded_quality"
        assert item.metadata == {"source": "test"}
        assert item.reviewer == "alice"

    def test_from_row_empty_metadata(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            "CREATE TABLE t ("
            "review_id TEXT, job_id TEXT, reason TEXT, confidence REAL, "
            "quality_classification TEXT, status TEXT, reviewer TEXT, "
            "decision_notes TEXT, created_at TEXT, reviewed_at TEXT, "
            "metadata_json TEXT)"
        )
        conn.execute(
            "INSERT INTO t VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("rev_a", "job_a", "low_confidence", 0.0, "", "pending", "", "", "", "", ""),
        )
        row = conn.execute("SELECT * FROM t").fetchone()
        item = ReviewItem.from_row(row)
        assert item.metadata == {}

    def test_from_row_invalid_json_metadata(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            "CREATE TABLE t ("
            "review_id TEXT, job_id TEXT, reason TEXT, confidence REAL, "
            "quality_classification TEXT, status TEXT, reviewer TEXT, "
            "decision_notes TEXT, created_at TEXT, reviewed_at TEXT, "
            "metadata_json TEXT)"
        )
        conn.execute(
            "INSERT INTO t VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("rev_a", "job_a", "low_confidence", 0.0, "", "pending", "", "", "", "", "{bad json"),
        )
        row = conn.execute("SELECT * FROM t").fetchone()
        item = ReviewItem.from_row(row)
        assert item.metadata == {}


# ---------------------------------------------------------------------------
# ReviewQueue.add()
# ---------------------------------------------------------------------------


class TestReviewQueueAdd:
    def test_add_creates_item(self, queue):
        item = queue.add(
            job_id="job_aabbccddee01",
            reason=ReviewReason.LOW_CONFIDENCE,
            confidence=0.35,
            quality_classification="review_required",
        )
        assert item.review_id.startswith("rev_")
        assert len(item.review_id) == 16  # "rev_" + 12 hex chars
        assert item.job_id == "job_aabbccddee01"
        assert item.reason == "low_confidence"
        assert item.confidence == 0.35
        assert item.status == ReviewStatus.PENDING
        assert item.created_at != ""

    def test_add_with_metadata(self, queue):
        meta = {"pages_image_only": 3, "methods": ["PaddleOCR", "ImageOnly"]}
        item = queue.add(
            job_id="job_aabbccddee02",
            reason=ReviewReason.IMAGE_ONLY_PAGES,
            metadata=meta,
        )
        assert item.metadata == meta

    def test_add_stores_in_database(self, queue):
        item = queue.add(
            job_id="job_aabbccddee03",
            reason=ReviewReason.DEGRADED_QUALITY,
            confidence=0.45,
            quality_classification="degraded",
        )
        fetched = queue.get(item.review_id)
        assert fetched is not None
        assert fetched.job_id == "job_aabbccddee03"
        assert fetched.confidence == 0.45

    def test_add_multiple_items(self, queue):
        for i in range(5):
            queue.add(
                job_id=f"job_{i:012x}",
                reason=ReviewReason.LOW_CONFIDENCE,
                confidence=0.1 * i,
            )
        items = queue.list_pending()
        assert len(items) == 5


# ---------------------------------------------------------------------------
# ReviewQueue.get()
# ---------------------------------------------------------------------------


class TestReviewQueueGet:
    def test_get_existing_item(self, queue):
        item = queue.add(job_id="job_aabbccddee04", reason="manual_flag")
        fetched = queue.get(item.review_id)
        assert fetched is not None
        assert fetched.review_id == item.review_id

    def test_get_nonexistent_returns_none(self, queue):
        assert queue.get("rev_doesnotexist") is None


# ---------------------------------------------------------------------------
# ReviewQueue.list_pending()
# ---------------------------------------------------------------------------


class TestReviewQueueListPending:
    def test_list_pending_returns_pending_only(self, queue):
        item1 = queue.add(job_id="job_000000000001", reason="low_confidence")
        item2 = queue.add(job_id="job_000000000002", reason="degraded_quality")
        # Decide on item1
        queue.decide(item1.review_id, ReviewStatus.APPROVED, reviewer="tester")

        pending = queue.list_pending()
        assert len(pending) == 1
        assert pending[0].review_id == item2.review_id

    def test_list_pending_with_reason_filter(self, queue):
        queue.add(job_id="job_000000000001", reason="low_confidence")
        queue.add(job_id="job_000000000002", reason="degraded_quality")
        queue.add(job_id="job_000000000003", reason="low_confidence")

        filtered = queue.list_pending(reason="low_confidence")
        assert len(filtered) == 2
        assert all(i.reason == "low_confidence" for i in filtered)

    def test_list_pending_with_limit_and_offset(self, queue):
        for i in range(10):
            queue.add(job_id=f"job_{i:012x}", reason="low_confidence")

        page1 = queue.list_pending(limit=3, offset=0)
        page2 = queue.list_pending(limit=3, offset=3)
        assert len(page1) == 3
        assert len(page2) == 3
        # No overlap
        ids1 = {i.review_id for i in page1}
        ids2 = {i.review_id for i in page2}
        assert ids1.isdisjoint(ids2)

    def test_list_pending_empty_queue(self, queue):
        assert queue.list_pending() == []


# ---------------------------------------------------------------------------
# ReviewQueue.decide()
# ---------------------------------------------------------------------------


class TestReviewQueueDecide:
    def test_decide_approve(self, queue):
        item = queue.add(job_id="job_000000000001", reason="low_confidence")
        result = queue.decide(
            item.review_id, ReviewStatus.APPROVED, reviewer="alice", notes="OK"
        )
        assert result is not None
        assert result.status == ReviewStatus.APPROVED
        assert result.reviewer == "alice"
        assert result.decision_notes == "OK"
        assert result.reviewed_at != ""

    def test_decide_reject(self, queue):
        item = queue.add(job_id="job_000000000001", reason="low_confidence")
        result = queue.decide(item.review_id, ReviewStatus.REJECTED, reviewer="bob")
        assert result.status == ReviewStatus.REJECTED

    def test_decide_reprocess(self, queue):
        item = queue.add(job_id="job_000000000001", reason="dpi_escalation_failed")
        result = queue.decide(
            item.review_id, ReviewStatus.REPROCESS, notes="Try higher DPI"
        )
        assert result.status == ReviewStatus.REPROCESS
        assert result.decision_notes == "Try higher DPI"

    def test_decide_nonexistent_returns_none(self, queue):
        result = queue.decide("rev_doesnotexist", ReviewStatus.APPROVED)
        assert result is None

    def test_decide_already_decided_raises(self, queue):
        item = queue.add(job_id="job_000000000001", reason="low_confidence")
        queue.decide(item.review_id, ReviewStatus.APPROVED, reviewer="alice")
        with pytest.raises(ValueError, match="already"):
            queue.decide(item.review_id, ReviewStatus.REJECTED, reviewer="bob")

    def test_decide_invalid_status_raises(self, queue):
        item = queue.add(job_id="job_000000000001", reason="low_confidence")
        with pytest.raises(ValueError, match="Invalid decision status"):
            queue.decide(item.review_id, "pending")

    def test_decide_persists_to_database(self, queue):
        item = queue.add(job_id="job_000000000001", reason="low_confidence")
        queue.decide(item.review_id, ReviewStatus.APPROVED, reviewer="eve")
        fetched = queue.get(item.review_id)
        assert fetched.status == ReviewStatus.APPROVED
        assert fetched.reviewer == "eve"


# ---------------------------------------------------------------------------
# ReviewQueue.stats()
# ---------------------------------------------------------------------------


class TestReviewQueueStats:
    def test_stats_empty_queue(self, queue):
        stats = queue.stats()
        assert stats["pending"] == 0
        assert stats["approved"] == 0
        assert stats["rejected"] == 0
        assert stats["reprocess"] == 0
        assert stats["total"] == 0
        assert stats["oldest_pending"] == ""

    def test_stats_correct_counts(self, queue):
        i1 = queue.add(job_id="job_000000000001", reason="low_confidence")
        i2 = queue.add(job_id="job_000000000002", reason="degraded_quality")
        queue.add(job_id="job_000000000003", reason="low_confidence")
        queue.add(job_id="job_000000000004", reason="manual_flag")

        queue.decide(i1.review_id, ReviewStatus.APPROVED, reviewer="a")
        queue.decide(i2.review_id, ReviewStatus.REJECTED, reviewer="b")

        stats = queue.stats()
        assert stats["pending"] == 2
        assert stats["approved"] == 1
        assert stats["rejected"] == 1
        assert stats["reprocess"] == 0
        assert stats["total"] == 4

    def test_stats_oldest_pending(self, queue):
        i1 = queue.add(job_id="job_000000000001", reason="low_confidence")
        queue.add(job_id="job_000000000002", reason="low_confidence")

        stats = queue.stats()
        assert stats["oldest_pending"] == i1.created_at

    def test_stats_avg_review_time(self, queue):
        item = queue.add(job_id="job_000000000001", reason="low_confidence")
        queue.decide(item.review_id, ReviewStatus.APPROVED, reviewer="a")
        stats = queue.stats()
        # avg_review_seconds should be a number >= 0
        assert isinstance(stats["avg_review_seconds"], float)
        assert stats["avg_review_seconds"] >= 0.0


# ---------------------------------------------------------------------------
# ReviewQueue.list_all() and count()
# ---------------------------------------------------------------------------


class TestReviewQueueListAll:
    def test_list_all_no_filters(self, queue):
        queue.add(job_id="job_000000000001", reason="low_confidence")
        i2 = queue.add(job_id="job_000000000002", reason="degraded_quality")
        queue.decide(i2.review_id, ReviewStatus.APPROVED, reviewer="a")

        all_items = queue.list_all()
        assert len(all_items) == 2

    def test_list_all_filter_by_status(self, queue):
        queue.add(job_id="job_000000000001", reason="low_confidence")
        i2 = queue.add(job_id="job_000000000002", reason="degraded_quality")
        queue.decide(i2.review_id, ReviewStatus.APPROVED, reviewer="a")

        approved = queue.list_all(status=ReviewStatus.APPROVED)
        assert len(approved) == 1
        assert approved[0].status == ReviewStatus.APPROVED

    def test_list_all_filter_by_reason(self, queue):
        queue.add(job_id="job_000000000001", reason="low_confidence")
        queue.add(job_id="job_000000000002", reason="degraded_quality")
        queue.add(job_id="job_000000000003", reason="low_confidence")

        lc_items = queue.list_all(reason="low_confidence")
        assert len(lc_items) == 2

    def test_list_all_combined_filters(self, queue):
        i1 = queue.add(job_id="job_000000000001", reason="low_confidence")
        queue.add(job_id="job_000000000002", reason="degraded_quality")
        queue.add(job_id="job_000000000003", reason="low_confidence")
        queue.decide(i1.review_id, ReviewStatus.APPROVED, reviewer="a")

        result = queue.list_all(status=ReviewStatus.PENDING, reason="low_confidence")
        assert len(result) == 1

    def test_count_matches_list(self, queue):
        for i in range(7):
            queue.add(job_id=f"job_{i:012x}", reason="low_confidence")
        assert queue.count() == 7
        assert queue.count(status=ReviewStatus.PENDING) == 7
        assert queue.count(reason="low_confidence") == 7
        assert queue.count(reason="degraded_quality") == 0


# ---------------------------------------------------------------------------
# Enum values
# ---------------------------------------------------------------------------


class TestEnums:
    def test_review_status_values(self):
        assert ReviewStatus.PENDING == "pending"
        assert ReviewStatus.APPROVED == "approved"
        assert ReviewStatus.REJECTED == "rejected"
        assert ReviewStatus.REPROCESS == "reprocess"

    def test_review_reason_values(self):
        assert ReviewReason.LOW_CONFIDENCE == "low_confidence"
        assert ReviewReason.DEGRADED_QUALITY == "degraded_quality"
        assert ReviewReason.HANDWRITING_DETECTED == "handwriting_detected"
        assert ReviewReason.DPI_ESCALATION_FAILED == "dpi_escalation_failed"
        assert ReviewReason.IMAGE_ONLY_PAGES == "image_only_pages"
        assert ReviewReason.CLASSIFICATION_UNCERTAIN == "classification_uncertain"
        assert ReviewReason.MANUAL_FLAG == "manual_flag"
