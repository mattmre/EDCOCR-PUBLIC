"""
Unit tests for historical processing analytics (api/analytics.py).

Tests cover:
- TimeGranularity enum values and count
- JobRecord creation and fields
- PeriodStats defaults and to_dict keys
- TrendAnalysis defaults and to_dict structure
- AnalyticsStore construction
- record_job stores records
- record_job with explicit timestamp
- record_job max_records eviction
- get_period_stats empty period
- get_period_stats aggregation
- get_period_stats duration percentiles
- get_period_stats engine/language/doc_type breakdown
- get_period_stats success_rate calculation
- get_period_stats pages_per_minute
- get_time_series hourly bucketing
- get_time_series daily bucketing
- get_time_series empty range
- get_trend with data
- get_trend with no previous data
- get_top_engines ranking
- get_top_languages ranking
- get_worker_stats aggregation
- get_worker_stats empty
- record_count property
- reset clears all
- get_analytics_store singleton
- Thread safety concurrent operations

Run with: python -m pytest tests/test_analytics.py -v
"""

import threading
import time

# Add project root to path
from api.analytics import (
    AnalyticsStore,
    JobRecord,
    PeriodStats,
    TimeGranularity,
    TrendAnalysis,
    get_analytics_store,
)

# ---------------------------------------------------------------------------
# Tests: TimeGranularity
# ---------------------------------------------------------------------------


class TestTimeGranularity:
    def test_enum_values(self):
        assert TimeGranularity.HOURLY.value == "hourly"
        assert TimeGranularity.DAILY.value == "daily"
        assert TimeGranularity.WEEKLY.value == "weekly"

    def test_enum_count(self):
        assert len(TimeGranularity) == 3


# ---------------------------------------------------------------------------
# Tests: JobRecord
# ---------------------------------------------------------------------------


class TestJobRecord:
    def test_creation(self):
        r = JobRecord(job_id="j-1", timestamp=1000.0)
        assert r.job_id == "j-1"
        assert r.timestamp == 1000.0
        assert r.pages == 0
        assert r.duration_seconds == 0.0
        assert r.file_size_bytes == 0
        assert r.success is True
        assert r.engine == ""
        assert r.language == ""
        assert r.doc_type == ""
        assert r.worker_id == ""

    def test_creation_with_all_fields(self):
        r = JobRecord(
            job_id="j-2",
            timestamp=2000.0,
            pages=5,
            duration_seconds=12.5,
            file_size_bytes=1024000,
            success=False,
            engine="paddle",
            language="en",
            doc_type="invoice",
            worker_id="w-1",
        )
        assert r.job_id == "j-2"
        assert r.pages == 5
        assert r.duration_seconds == 12.5
        assert r.file_size_bytes == 1024000
        assert r.success is False
        assert r.engine == "paddle"
        assert r.language == "en"
        assert r.doc_type == "invoice"
        assert r.worker_id == "w-1"


# ---------------------------------------------------------------------------
# Tests: PeriodStats
# ---------------------------------------------------------------------------


class TestPeriodStats:
    def test_defaults(self):
        s = PeriodStats()
        assert s.period_start == 0.0
        assert s.period_end == 0.0
        assert s.total_jobs == 0
        assert s.successful_jobs == 0
        assert s.failed_jobs == 0
        assert s.total_pages == 0
        assert s.total_bytes == 0
        assert s.avg_duration_seconds == 0.0
        assert s.p50_duration_seconds == 0.0
        assert s.p95_duration_seconds == 0.0
        assert s.avg_pages_per_job == 0.0
        assert s.pages_per_minute == 0.0
        assert s.success_rate == 0.0
        assert s.engine_breakdown == {}
        assert s.language_breakdown == {}
        assert s.doc_type_breakdown == {}

    def test_to_dict_keys(self):
        d = PeriodStats().to_dict()
        expected_keys = {
            "period_start", "period_end", "total_jobs",
            "successful_jobs", "failed_jobs", "total_pages",
            "total_bytes", "avg_duration_seconds",
            "p50_duration_seconds", "p95_duration_seconds",
            "avg_pages_per_job", "pages_per_minute",
            "success_rate", "engine_breakdown",
            "language_breakdown", "doc_type_breakdown",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_rounding(self):
        s = PeriodStats(
            avg_duration_seconds=1.23456,
            p50_duration_seconds=0.98765,
            p95_duration_seconds=5.55555,
            avg_pages_per_job=3.33333,
            pages_per_minute=12.34567,
            success_rate=0.98765,
        )
        d = s.to_dict()
        assert d["avg_duration_seconds"] == 1.23
        assert d["p50_duration_seconds"] == 0.99
        assert d["p95_duration_seconds"] == 5.56
        assert d["avg_pages_per_job"] == 3.33
        assert d["pages_per_minute"] == 12.35
        assert d["success_rate"] == 0.9877

    def test_to_dict_breakdown_copy(self):
        s = PeriodStats(engine_breakdown={"paddle": 10})
        d = s.to_dict()
        d["engine_breakdown"]["tesseract"] = 5
        # Original should not be affected
        assert "tesseract" not in s.engine_breakdown


# ---------------------------------------------------------------------------
# Tests: TrendAnalysis
# ---------------------------------------------------------------------------


class TestTrendAnalysis:
    def test_defaults(self):
        t = TrendAnalysis()
        assert isinstance(t.current, PeriodStats)
        assert isinstance(t.previous, PeriodStats)
        assert t.throughput_change_pct == 0.0
        assert t.latency_change_pct == 0.0
        assert t.success_rate_change_pct == 0.0
        assert t.volume_change_pct == 0.0

    def test_to_dict_structure(self):
        t = TrendAnalysis()
        d = t.to_dict()
        assert "current" in d
        assert "previous" in d
        assert "changes" in d
        assert "throughput_pct" in d["changes"]
        assert "latency_pct" in d["changes"]
        assert "success_rate_pct" in d["changes"]
        assert "volume_pct" in d["changes"]

    def test_to_dict_values(self):
        t = TrendAnalysis(
            throughput_change_pct=15.556,
            latency_change_pct=-8.333,
            success_rate_change_pct=2.111,
            volume_change_pct=50.999,
        )
        d = t.to_dict()
        assert d["changes"]["throughput_pct"] == 15.56
        assert d["changes"]["latency_pct"] == -8.33
        assert d["changes"]["success_rate_pct"] == 2.11
        assert d["changes"]["volume_pct"] == 51.0

    def test_to_dict_nested_period_stats(self):
        t = TrendAnalysis(
            current=PeriodStats(total_jobs=10),
            previous=PeriodStats(total_jobs=5),
        )
        d = t.to_dict()
        assert d["current"]["total_jobs"] == 10
        assert d["previous"]["total_jobs"] == 5


# ---------------------------------------------------------------------------
# Tests: AnalyticsStore construction
# ---------------------------------------------------------------------------


class TestAnalyticsStoreConstruction:
    def test_construction(self):
        store = AnalyticsStore()
        assert store.record_count == 0

    def test_custom_max_records(self):
        store = AnalyticsStore(max_records=50)
        assert store.record_count == 0


# ---------------------------------------------------------------------------
# Tests: record_job
# ---------------------------------------------------------------------------


class TestRecordJob:
    def test_stores_record(self):
        store = AnalyticsStore()
        store.record_job("j-1", pages=3, engine="paddle")
        assert store.record_count == 1

    def test_multiple_records(self):
        store = AnalyticsStore()
        store.record_job("j-1")
        store.record_job("j-2")
        store.record_job("j-3")
        assert store.record_count == 3

    def test_explicit_timestamp(self):
        store = AnalyticsStore()
        store.record_job("j-1", timestamp=5000.0, pages=2)
        stats = store.get_period_stats(4999.0, 5001.0)
        assert stats.total_jobs == 1
        assert stats.total_pages == 2

    def test_auto_timestamp(self):
        store = AnalyticsStore()
        before = time.time()
        store.record_job("j-1")
        after = time.time()
        stats = store.get_period_stats(before - 1, after + 1)
        assert stats.total_jobs == 1

    def test_max_records_eviction(self):
        store = AnalyticsStore(max_records=5)
        for i in range(10):
            store.record_job(f"j-{i}", timestamp=float(i))
        assert store.record_count == 5
        # Oldest records (0-4) should be evicted; latest (5-9) remain
        stats = store.get_period_stats(0.0, 5.0)
        assert stats.total_jobs == 0
        stats = store.get_period_stats(5.0, 10.0)
        assert stats.total_jobs == 5


# ---------------------------------------------------------------------------
# Tests: get_period_stats
# ---------------------------------------------------------------------------


class TestGetPeriodStats:
    def test_empty_period(self):
        store = AnalyticsStore()
        stats = store.get_period_stats(0.0, 100.0)
        assert stats.total_jobs == 0
        assert stats.successful_jobs == 0
        assert stats.failed_jobs == 0
        assert stats.total_pages == 0
        assert stats.total_bytes == 0
        assert stats.avg_duration_seconds == 0.0
        assert stats.success_rate == 0.0

    def test_empty_period_preserves_bounds(self):
        store = AnalyticsStore()
        stats = store.get_period_stats(100.0, 200.0)
        assert stats.period_start == 100.0
        assert stats.period_end == 200.0

    def test_aggregation(self):
        store = AnalyticsStore()
        store.record_job("j-1", timestamp=10.0, pages=3, file_size_bytes=1000, duration_seconds=2.0)
        store.record_job("j-2", timestamp=20.0, pages=5, file_size_bytes=2000, duration_seconds=4.0)
        store.record_job("j-3", timestamp=30.0, pages=2, file_size_bytes=500, duration_seconds=1.0)

        stats = store.get_period_stats(0.0, 50.0)
        assert stats.total_jobs == 3
        assert stats.successful_jobs == 3
        assert stats.failed_jobs == 0
        assert stats.total_pages == 10
        assert stats.total_bytes == 3500

    def test_filters_by_time_range(self):
        store = AnalyticsStore()
        store.record_job("j-1", timestamp=10.0, pages=1)
        store.record_job("j-2", timestamp=20.0, pages=2)
        store.record_job("j-3", timestamp=30.0, pages=3)

        stats = store.get_period_stats(15.0, 25.0)
        assert stats.total_jobs == 1
        assert stats.total_pages == 2

    def test_end_exclusive(self):
        store = AnalyticsStore()
        store.record_job("j-1", timestamp=10.0)
        # end is exclusive: timestamp == end should not be included
        stats = store.get_period_stats(0.0, 10.0)
        assert stats.total_jobs == 0
        # start is inclusive: timestamp == start should be included
        stats = store.get_period_stats(10.0, 20.0)
        assert stats.total_jobs == 1

    def test_duration_percentiles(self):
        store = AnalyticsStore()
        # Create 100 jobs with durations 1..100
        for i in range(1, 101):
            store.record_job(f"j-{i}", timestamp=float(i), duration_seconds=float(i))

        stats = store.get_period_stats(0.0, 200.0)
        # p50 should be around 50
        assert 49 <= stats.p50_duration_seconds <= 51
        # p95 should be around 95
        assert 94 <= stats.p95_duration_seconds <= 96

    def test_duration_percentiles_single_job(self):
        store = AnalyticsStore()
        store.record_job("j-1", timestamp=1.0, duration_seconds=7.5)
        stats = store.get_period_stats(0.0, 10.0)
        assert stats.p50_duration_seconds == 7.5
        assert stats.p95_duration_seconds == 7.5

    def test_duration_zero_excluded(self):
        store = AnalyticsStore()
        store.record_job("j-1", timestamp=1.0, duration_seconds=0.0)
        store.record_job("j-2", timestamp=2.0, duration_seconds=10.0)
        stats = store.get_period_stats(0.0, 10.0)
        # avg should be based only on j-2 (duration > 0)
        assert stats.avg_duration_seconds == 10.0

    def test_engine_breakdown(self):
        store = AnalyticsStore()
        store.record_job("j-1", timestamp=1.0, engine="paddle")
        store.record_job("j-2", timestamp=2.0, engine="paddle")
        store.record_job("j-3", timestamp=3.0, engine="tesseract")
        store.record_job("j-4", timestamp=4.0, engine="")  # no engine

        stats = store.get_period_stats(0.0, 10.0)
        assert stats.engine_breakdown == {"paddle": 2, "tesseract": 1}

    def test_language_breakdown(self):
        store = AnalyticsStore()
        store.record_job("j-1", timestamp=1.0, language="en")
        store.record_job("j-2", timestamp=2.0, language="en")
        store.record_job("j-3", timestamp=3.0, language="de")

        stats = store.get_period_stats(0.0, 10.0)
        assert stats.language_breakdown == {"en": 2, "de": 1}

    def test_doc_type_breakdown(self):
        store = AnalyticsStore()
        store.record_job("j-1", timestamp=1.0, doc_type="invoice")
        store.record_job("j-2", timestamp=2.0, doc_type="invoice")
        store.record_job("j-3", timestamp=3.0, doc_type="receipt")
        store.record_job("j-4", timestamp=4.0, doc_type="contract")

        stats = store.get_period_stats(0.0, 10.0)
        assert stats.doc_type_breakdown == {"invoice": 2, "receipt": 1, "contract": 1}

    def test_success_rate(self):
        store = AnalyticsStore()
        store.record_job("j-1", timestamp=1.0, success=True)
        store.record_job("j-2", timestamp=2.0, success=True)
        store.record_job("j-3", timestamp=3.0, success=False)
        store.record_job("j-4", timestamp=4.0, success=True)

        stats = store.get_period_stats(0.0, 10.0)
        assert stats.successful_jobs == 3
        assert stats.failed_jobs == 1
        assert stats.success_rate == 0.75

    def test_success_rate_all_failed(self):
        store = AnalyticsStore()
        store.record_job("j-1", timestamp=1.0, success=False)
        store.record_job("j-2", timestamp=2.0, success=False)
        stats = store.get_period_stats(0.0, 10.0)
        assert stats.success_rate == 0.0

    def test_pages_per_minute(self):
        store = AnalyticsStore()
        # 60-second window with 120 pages => 2 pages/sec => 120 pages/min
        store.record_job("j-1", timestamp=10.0, pages=60)
        store.record_job("j-2", timestamp=50.0, pages=60)

        stats = store.get_period_stats(0.0, 60.0)
        # 120 pages in 1 minute
        assert stats.pages_per_minute == 120.0

    def test_avg_pages_per_job(self):
        store = AnalyticsStore()
        store.record_job("j-1", timestamp=1.0, pages=4)
        store.record_job("j-2", timestamp=2.0, pages=6)
        store.record_job("j-3", timestamp=3.0, pages=8)

        stats = store.get_period_stats(0.0, 10.0)
        assert stats.avg_pages_per_job == 6.0


# ---------------------------------------------------------------------------
# Tests: get_time_series
# ---------------------------------------------------------------------------


class TestGetTimeSeries:
    def test_hourly_bucketing(self):
        store = AnalyticsStore()
        # Insert jobs across 3 hours
        store.record_job("j-1", timestamp=100.0, pages=1)         # bucket 0-3600
        store.record_job("j-2", timestamp=3700.0, pages=2)        # bucket 3600-7200
        store.record_job("j-3", timestamp=7300.0, pages=3)        # bucket 7200-10800

        series = store.get_time_series(0.0, 10800.0, TimeGranularity.HOURLY)
        assert len(series) == 3
        assert series[0].total_pages == 1
        assert series[1].total_pages == 2
        assert series[2].total_pages == 3

    def test_daily_bucketing(self):
        store = AnalyticsStore()
        store.record_job("j-1", timestamp=100.0, pages=5)          # day 1
        store.record_job("j-2", timestamp=86500.0, pages=10)       # day 2

        series = store.get_time_series(0.0, 172800.0, TimeGranularity.DAILY)
        assert len(series) == 2
        assert series[0].total_pages == 5
        assert series[1].total_pages == 10

    def test_empty_range(self):
        store = AnalyticsStore()
        series = store.get_time_series(0.0, 3600.0, TimeGranularity.HOURLY)
        assert len(series) == 1
        assert series[0].total_jobs == 0

    def test_partial_bucket(self):
        store = AnalyticsStore()
        store.record_job("j-1", timestamp=100.0, pages=1)
        # Range doesn't align to full hour
        series = store.get_time_series(0.0, 1800.0, TimeGranularity.HOURLY)
        assert len(series) == 1
        assert series[0].total_pages == 1

    def test_weekly_bucketing(self):
        store = AnalyticsStore()
        store.record_job("j-1", timestamp=100.0, pages=1)
        store.record_job("j-2", timestamp=604900.0, pages=2)  # second week

        series = store.get_time_series(0.0, 1209600.0, TimeGranularity.WEEKLY)
        assert len(series) == 2
        assert series[0].total_pages == 1
        assert series[1].total_pages == 2


# ---------------------------------------------------------------------------
# Tests: get_trend
# ---------------------------------------------------------------------------


class TestGetTrend:
    def test_with_data(self):
        store = AnalyticsStore()
        now = time.time()
        # Current period: last 60 seconds
        for i in range(10):
            store.record_job(f"curr-{i}", timestamp=now - 30 + i, pages=5, duration_seconds=2.0)
        # Previous period: 60-120 seconds ago
        for i in range(5):
            store.record_job(f"prev-{i}", timestamp=now - 90 + i, pages=3, duration_seconds=4.0)

        trend = store.get_trend(period_seconds=60)
        assert trend.current.total_jobs == 10
        assert trend.previous.total_jobs == 5
        assert trend.volume_change_pct == 100.0  # doubled

    def test_with_no_previous_data(self):
        store = AnalyticsStore()
        now = time.time()
        store.record_job("j-1", timestamp=now - 10, pages=5)

        trend = store.get_trend(period_seconds=60)
        assert trend.current.total_jobs == 1
        assert trend.previous.total_jobs == 0
        # No previous data means changes stay 0
        assert trend.throughput_change_pct == 0.0
        assert trend.latency_change_pct == 0.0
        assert trend.success_rate_change_pct == 0.0
        assert trend.volume_change_pct == 0.0

    def test_trend_to_dict(self):
        store = AnalyticsStore()
        trend = store.get_trend(period_seconds=60)
        d = trend.to_dict()
        assert "current" in d
        assert "previous" in d
        assert "changes" in d


# ---------------------------------------------------------------------------
# Tests: get_top_engines
# ---------------------------------------------------------------------------


class TestGetTopEngines:
    def test_ranking(self):
        store = AnalyticsStore()
        for i in range(10):
            store.record_job(f"p-{i}", timestamp=float(i + 1), engine="paddle")
        for i in range(7):
            store.record_job(f"t-{i}", timestamp=float(i + 1), engine="tesseract")
        for i in range(3):
            store.record_job(f"e-{i}", timestamp=float(i + 1), engine="easyocr")

        top = store.get_top_engines(0.0, 100.0)
        assert len(top) == 3
        assert top[0] == ("paddle", 10)
        assert top[1] == ("tesseract", 7)
        assert top[2] == ("easyocr", 3)

    def test_limit(self):
        store = AnalyticsStore()
        for i in range(5):
            store.record_job(f"j-{i}", timestamp=float(i + 1), engine=f"engine-{i}")
        top = store.get_top_engines(0.0, 100.0, limit=2)
        assert len(top) == 2

    def test_empty(self):
        store = AnalyticsStore()
        top = store.get_top_engines(0.0, 100.0)
        assert top == []


# ---------------------------------------------------------------------------
# Tests: get_top_languages
# ---------------------------------------------------------------------------


class TestGetTopLanguages:
    def test_ranking(self):
        store = AnalyticsStore()
        for i in range(8):
            store.record_job(f"en-{i}", timestamp=float(i + 1), language="en")
        for i in range(5):
            store.record_job(f"de-{i}", timestamp=float(i + 1), language="de")
        for i in range(2):
            store.record_job(f"fr-{i}", timestamp=float(i + 1), language="fr")

        top = store.get_top_languages(0.0, 100.0)
        assert len(top) == 3
        assert top[0] == ("en", 8)
        assert top[1] == ("de", 5)
        assert top[2] == ("fr", 2)

    def test_limit(self):
        store = AnalyticsStore()
        for i in range(5):
            store.record_job(f"j-{i}", timestamp=float(i + 1), language=f"lang-{i}")
        top = store.get_top_languages(0.0, 100.0, limit=1)
        assert len(top) == 1

    def test_empty(self):
        store = AnalyticsStore()
        top = store.get_top_languages(0.0, 100.0)
        assert top == []


# ---------------------------------------------------------------------------
# Tests: get_worker_stats
# ---------------------------------------------------------------------------


class TestGetWorkerStats:
    def test_aggregation(self):
        store = AnalyticsStore()
        store.record_job("j-1", timestamp=1.0, worker_id="w-1", pages=3, duration_seconds=2.0, success=True)
        store.record_job("j-2", timestamp=2.0, worker_id="w-1", pages=5, duration_seconds=4.0, success=True)
        store.record_job("j-3", timestamp=3.0, worker_id="w-1", pages=2, duration_seconds=6.0, success=False)
        store.record_job("j-4", timestamp=4.0, worker_id="w-2", pages=10, duration_seconds=3.0, success=True)

        ws = store.get_worker_stats(0.0, 10.0)
        assert "w-1" in ws
        assert "w-2" in ws

        w1 = ws["w-1"]
        assert w1["worker_id"] == "w-1"
        assert w1["total_jobs"] == 3
        assert w1["total_pages"] == 10
        assert w1["successes"] == 2
        assert w1["failures"] == 1
        assert w1["avg_duration"] == 4.0
        assert w1["success_rate"] == round(2 / 3, 4)

        w2 = ws["w-2"]
        assert w2["total_jobs"] == 1
        assert w2["total_pages"] == 10
        assert w2["successes"] == 1
        assert w2["failures"] == 0
        assert w2["avg_duration"] == 3.0
        assert w2["success_rate"] == 1.0

    def test_empty(self):
        store = AnalyticsStore()
        ws = store.get_worker_stats(0.0, 100.0)
        assert ws == {}

    def test_excludes_no_worker_id(self):
        store = AnalyticsStore()
        store.record_job("j-1", timestamp=1.0, worker_id="")
        store.record_job("j-2", timestamp=2.0, worker_id="w-1")
        ws = store.get_worker_stats(0.0, 10.0)
        assert len(ws) == 1
        assert "w-1" in ws

    def test_filters_by_time_range(self):
        store = AnalyticsStore()
        store.record_job("j-1", timestamp=5.0, worker_id="w-1")
        store.record_job("j-2", timestamp=15.0, worker_id="w-1")
        ws = store.get_worker_stats(10.0, 20.0)
        assert ws["w-1"]["total_jobs"] == 1


# ---------------------------------------------------------------------------
# Tests: record_count property
# ---------------------------------------------------------------------------


class TestRecordCount:
    def test_initial(self):
        store = AnalyticsStore()
        assert store.record_count == 0

    def test_after_inserts(self):
        store = AnalyticsStore()
        store.record_job("j-1")
        store.record_job("j-2")
        assert store.record_count == 2


# ---------------------------------------------------------------------------
# Tests: reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_clears_all(self):
        store = AnalyticsStore()
        store.record_job("j-1", timestamp=1.0)
        store.record_job("j-2", timestamp=2.0)
        store.record_job("j-3", timestamp=3.0)
        assert store.record_count == 3
        store.reset()
        assert store.record_count == 0
        stats = store.get_period_stats(0.0, 100.0)
        assert stats.total_jobs == 0


# ---------------------------------------------------------------------------
# Tests: get_analytics_store singleton
# ---------------------------------------------------------------------------


class TestGetAnalyticsStore:
    def test_singleton(self):
        s1 = get_analytics_store()
        s2 = get_analytics_store()
        assert s1 is s2
        assert isinstance(s1, AnalyticsStore)


# ---------------------------------------------------------------------------
# Tests: Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_record_and_query(self):
        store = AnalyticsStore()
        errors = []
        barrier = threading.Barrier(4)

        def record_jobs(start_id, count):
            try:
                barrier.wait(timeout=5)
                for i in range(count):
                    store.record_job(
                        f"j-{start_id + i}",
                        timestamp=float(start_id + i),
                        pages=i + 1,
                        duration_seconds=float(i + 1),
                        engine="paddle",
                        language="en",
                        worker_id=f"w-{start_id}",
                    )
                    # Also query while recording
                    store.get_period_stats(0.0, 100000.0)
                    store.get_worker_stats(0.0, 100000.0)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=record_jobs, args=(0, 25)),
            threading.Thread(target=record_jobs, args=(100, 25)),
            threading.Thread(target=record_jobs, args=(200, 25)),
            threading.Thread(target=record_jobs, args=(300, 25)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == [], f"Thread errors: {errors}"
        assert store.record_count == 100
