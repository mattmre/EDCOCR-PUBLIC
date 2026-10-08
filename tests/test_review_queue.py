"""Tests for the review queue data model and storage layer."""

from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
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
            (
                "rev_a",
                "job_a",
                "low_confidence",
                0.0,
                "",
                "pending",
                "",
                "",
                "",
                "",
                "",
            ),
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
            (
                "rev_a",
                "job_a",
                "low_confidence",
                0.0,
                "",
                "pending",
                "",
                "",
                "",
                "",
                "{bad json",
            ),
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

    def test_add_unique_translation_review_is_idempotent(self, queue):
        metadata = {"batch_id": "batch-1", "tenant_id": "tenant-a"}
        first = queue.add_unique(
            "translation-batch:batch-1",
            ReviewReason.TRANSLATION_CERTIFICATION,
            metadata=metadata,
        )
        second = queue.add_unique(
            "translation-batch:batch-1",
            ReviewReason.TRANSLATION_CERTIFICATION,
            metadata=metadata,
        )
        assert second.review_id == first.review_id
        assert queue.count(reason=ReviewReason.TRANSLATION_CERTIFICATION) == 1
        certified = queue.certify(
            first.review_id,
            reviewer="reviewer-a",
            notes="verified",
            auth_method="oidc",
            custody_event_id="c" * 64,
        )
        assert certified.metadata["certified"] is True
        pending_promotions = queue.list_pending_translation_promotions()
        assert [candidate.review_id for candidate in pending_promotions] == [
            first.review_id
        ]
        assert queue.confirm_translation_promotion(first.review_id, "c" * 64)
        assert queue.list_pending_translation_promotions() == []
        recovered = queue.add_unique(
            "translation-batch:batch-1",
            ReviewReason.TRANSLATION_CERTIFICATION,
            metadata=metadata,
        )
        assert recovered.review_id == first.review_id
        assert recovered.metadata["certified"] is True
        with pytest.raises(ValueError, match="binding metadata changed"):
            queue.add_unique(
                "translation-batch:batch-1",
                ReviewReason.TRANSLATION_CERTIFICATION,
                metadata={"batch_id": "batch-1", "tenant_id": "tenant-b"},
            )

    def test_abandoned_translation_promotion_no_longer_blocks_newer_work(self, queue):
        reviews = []
        for batch_id in ("batch-orphan", "batch-live"):
            item = queue.add_unique(
                f"translation-batch:{batch_id}",
                ReviewReason.TRANSLATION_CERTIFICATION,
                metadata={"batch_id": batch_id, "tenant_id": "tenant-a"},
            )
            reviews.append(
                queue.certify(
                    item.review_id,
                    reviewer="reviewer-a",
                    notes="verified",
                    auth_method="oidc",
                    custody_event_id=("a" if batch_id == "batch-orphan" else "b") * 64,
                )
            )

        assert queue.abandon_translation_promotion(
            reviews[0].review_id,
            "coordinator batch no longer exists",
        )

        abandoned = queue.get(reviews[0].review_id)
        assert abandoned is not None
        assert abandoned.metadata["promotion_abandoned"] is True
        assert (
            abandoned.metadata["promotion_abandon_reason"]
            == "coordinator batch no longer exists"
        )
        assert [
            candidate.review_id
            for candidate in queue.list_pending_translation_promotions()
        ] == [reviews[1].review_id]

    def test_retryable_promotions_rotate_so_newer_work_cannot_starve(self, queue):
        reviews = []
        for index in range(101):
            batch_id = f"batch-fairness-{index:03d}"
            item = queue.add_unique(
                f"translation-batch:{batch_id}",
                ReviewReason.TRANSLATION_CERTIFICATION,
                metadata={"batch_id": batch_id, "tenant_id": "tenant-a"},
            )
            reviews.append(
                queue.certify(
                    item.review_id,
                    reviewer="reviewer-a",
                    notes="verified",
                    auth_method="oidc",
                    custody_event_id=f"{index:064x}",
                )
            )

        first_page = queue.list_pending_translation_promotions(limit=100)
        assert len(first_page) == 100
        assert reviews[-1].review_id not in {
            candidate.review_id for candidate in first_page
        }
        for candidate in first_page:
            assert queue.defer_translation_promotion(
                candidate.review_id,
                "batch has not completed yet",
            )

        next_page = queue.list_pending_translation_promotions(limit=100)
        assert reviews[-1].review_id in {candidate.review_id for candidate in next_page}
        deferred = queue.get(first_page[0].review_id)
        assert deferred is not None
        assert deferred.metadata["promotion_attempt_count"] == 1
        assert deferred.metadata["promotion_last_attempt_at"]
        assert deferred.metadata["promotion_last_error"] == (
            "batch has not completed yet"
        )

        assert queue.defer_translation_promotion(
            reviews[-1].review_id,
            "x" * 10_000,
        )
        bounded = queue.get(reviews[-1].review_id)
        assert bounded is not None
        assert len(bounded.metadata["promotion_last_error"]) == 4_096

    def test_add_unique_translation_review_handles_cross_instance_race(
        self, queue, monkeypatch
    ):
        other = ReviewQueue(db_path=queue._db_path)
        barrier = threading.Barrier(2)
        original_add = ReviewQueue.add

        def racing_add(instance, *args, **kwargs):
            barrier.wait(timeout=5)
            return original_add(instance, *args, **kwargs)

        monkeypatch.setattr(ReviewQueue, "add", racing_add)
        metadata = {"batch_id": "batch-race", "tenant_id": "tenant-a"}

        def add_and_report_transaction(instance):
            item = instance.add_unique(
                "translation-batch:batch-race",
                ReviewReason.TRANSLATION_CERTIFICATION,
                metadata=metadata,
            )
            return item.review_id, instance._get_conn().in_transaction

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(add_and_report_transaction, instance)
                for instance in (queue, other)
            ]
        outcomes = [future.result() for future in futures]
        review_ids = {review_id for review_id, _in_transaction in outcomes}
        assert len(review_ids) == 1
        assert all(not in_transaction for _review_id, in_transaction in outcomes)

    def test_certify_cross_instance_compare_and_swap_has_one_winner(
        self, queue, monkeypatch
    ):
        other = ReviewQueue(db_path=queue._db_path)
        item = queue.add_unique(
            "translation-batch:batch-certify-race",
            ReviewReason.TRANSLATION_CERTIFICATION,
            metadata={"batch_id": "batch-certify-race", "tenant_id": "tenant-a"},
        )
        barrier = threading.Barrier(2)
        original_from_row = ReviewItem.from_row.__func__

        def synchronized_from_row(cls, row):
            result = original_from_row(cls, row)
            if row["review_id"] == item.review_id and row["status"] == "pending":
                barrier.wait(timeout=5)
            return result

        monkeypatch.setattr(ReviewItem, "from_row", classmethod(synchronized_from_row))

        def certify(instance, reviewer, custody_id):
            return instance.certify(
                item.review_id,
                reviewer=reviewer,
                notes="verified",
                auth_method="oidc",
                custody_event_id=custody_id,
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(certify, queue, "reviewer-a", "a" * 64),
                pool.submit(certify, other, "reviewer-b", "b" * 64),
            ]
        outcomes = []
        for future in futures:
            try:
                outcomes.append(("success", future.result()))
            except ValueError as exc:
                outcomes.append(("lost", str(exc)))

        assert [kind for kind, _ in outcomes].count("success") == 1
        assert [kind for kind, _ in outcomes].count("lost") == 1
        stored = queue.get(item.review_id)
        winner = next(value for kind, value in outcomes if kind == "success")
        assert stored is not None
        assert stored.metadata["certification"] == winner.metadata["certification"]


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

    def test_stats_empty_allowed_job_ids_returns_zeros(self, queue):
        """Empty allowed_job_ids is fail-closed: zero stats, no global leak."""
        queue.add(job_id="job_000000000001", reason="low_confidence")
        stats = queue.stats(allowed_job_ids=[])
        assert stats["pending"] == 0
        assert stats["approved"] == 0
        assert stats["rejected"] == 0
        assert stats["reprocess"] == 0
        assert stats["total"] == 0
        assert stats["avg_review_seconds"] == 0.0
        assert stats["oldest_pending"] == ""

    def test_stats_filtered_by_allowed_job_ids(self, queue):
        """Non-empty allowed_job_ids restricts aggregates to those jobs only."""
        i1 = queue.add(job_id="job_aaaa00000001", reason="low_confidence")
        i2 = queue.add(job_id="job_bbbb00000002", reason="degraded_quality")
        queue.add(job_id="job_aaaa00000003", reason="low_confidence")
        queue.decide(i1.review_id, ReviewStatus.APPROVED, reviewer="a")
        queue.decide(i2.review_id, ReviewStatus.REJECTED, reviewer="b")

        stats = queue.stats(allowed_job_ids=["job_aaaa00000001", "job_aaaa00000003"])
        assert stats["pending"] == 1
        assert stats["approved"] == 1
        assert stats["rejected"] == 0
        assert stats["total"] == 2

        # Foreign-only filter excludes tenant A items
        stats_b = queue.stats(allowed_job_ids=["job_bbbb00000002"])
        assert stats_b["pending"] == 0
        assert stats_b["rejected"] == 1
        assert stats_b["total"] == 1


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

    def test_count_handles_large_allowed_set(self, queue):
        queue.add(job_id="job_target", reason="low_confidence")
        allowed = [f"job_{index}" for index in range(40_000)]
        allowed.append("job_target")

        assert queue.count(allowed_job_ids=allowed) == 1
        assert [item.job_id for item in queue.list_all(allowed_job_ids=allowed)] == [
            "job_target"
        ]


class TestReviewQueueTenantPurge:
    def test_remove_tenant_metadata_deletes_translation_batch_rows(self, queue):
        queue.add(
            job_id="translation-batch:batch-keep",
            reason=ReviewReason.TRANSLATION_CERTIFICATION.value,
            metadata={"batch_id": "batch-keep", "tenant_id": "tenant-b"},
        )
        queue.add(
            job_id="translation-batch:batch-gone",
            reason=ReviewReason.TRANSLATION_CERTIFICATION.value,
            metadata={"batch_id": "batch-gone", "tenant_id": "tenant-a"},
        )
        removed = queue.remove_tenant_metadata("tenant-a")
        assert removed == 1
        remaining = [item.job_id for item in queue.list_all()]
        assert remaining == ["translation-batch:batch-keep"]

    def test_remove_tenant_metadata_deletes_exception_routing_resume_rows(self, queue):
        queue.add(
            job_id="abcdef0123456789",
            reason=ReviewReason.LOW_CONFIDENCE.value,
            metadata={
                "exception_router": True,
                "tenant_id": "tenant-a",
                "api_job_id": "job_aaaaaaaaaaaa",
            },
        )
        queue.add(
            job_id="fedcba9876543210",
            reason=ReviewReason.LOW_CONFIDENCE.value,
            metadata={
                "exception_router": True,
                "tenant_id": "tenant-b",
                "api_job_id": "job_bbbbbbbbbbbb",
            },
        )
        removed = queue.remove_tenant_metadata("tenant-a")
        assert removed == 1
        remaining = [item.job_id for item in queue.list_all()]
        assert remaining == ["fedcba9876543210"]


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
