"""Tests for api/event_bus.py — unified event bus abstraction."""

from __future__ import annotations

import json
import sys
import threading
import time
from unittest.mock import patch

import pytest

from api.event_bus import (
    Event,
    EventBus,
    EventType,
    KafkaEventBus,
    LocalEventBus,
    SnsEventBus,
    create_event_bus,
)

# ---------------------------------------------------------------------------
# EventType enum
# ---------------------------------------------------------------------------


class TestEventType:
    """Tests for EventType enum values."""

    def test_enum_has_10_members(self):
        assert len(EventType) == 10

    def test_job_submitted_value(self):
        assert EventType.JOB_SUBMITTED.value == "job.submitted"

    def test_job_started_value(self):
        assert EventType.JOB_STARTED.value == "job.started"

    def test_job_completed_value(self):
        assert EventType.JOB_COMPLETED.value == "job.completed"

    def test_job_failed_value(self):
        assert EventType.JOB_FAILED.value == "job.failed"

    def test_job_cancelled_value(self):
        assert EventType.JOB_CANCELLED.value == "job.cancelled"

    def test_page_processed_value(self):
        assert EventType.PAGE_PROCESSED.value == "page.processed"

    def test_worker_online_value(self):
        assert EventType.WORKER_ONLINE.value == "worker.online"

    def test_worker_offline_value(self):
        assert EventType.WORKER_OFFLINE.value == "worker.offline"

    def test_alert_triggered_value(self):
        assert EventType.ALERT_TRIGGERED.value == "alert.triggered"

    def test_custom_value(self):
        assert EventType.CUSTOM.value == "custom"


# ---------------------------------------------------------------------------
# Event dataclass
# ---------------------------------------------------------------------------


class TestEventCreation:
    """Tests for Event construction and auto-populated fields."""

    def test_event_requires_event_type(self):
        event = Event(event_type=EventType.JOB_SUBMITTED)
        assert event.event_type == EventType.JOB_SUBMITTED

    def test_auto_timestamp(self):
        before = time.time()
        event = Event(event_type=EventType.JOB_STARTED)
        after = time.time()
        assert before <= event.timestamp <= after

    def test_explicit_timestamp_preserved(self):
        event = Event(event_type=EventType.JOB_STARTED, timestamp=1234567890.0)
        assert event.timestamp == 1234567890.0

    def test_auto_event_id(self):
        event = Event(event_type=EventType.JOB_COMPLETED)
        assert event.event_id  # non-empty
        assert len(event.event_id) == 36  # UUID4 string

    def test_explicit_event_id_preserved(self):
        event = Event(event_type=EventType.JOB_COMPLETED, event_id="my-id-123")
        assert event.event_id == "my-id-123"

    def test_default_payload_is_empty_dict(self):
        event = Event(event_type=EventType.JOB_FAILED)
        assert event.payload == {}

    def test_payload_stored(self):
        event = Event(event_type=EventType.JOB_FAILED, payload={"error": "timeout"})
        assert event.payload == {"error": "timeout"}

    def test_default_source_is_empty(self):
        event = Event(event_type=EventType.CUSTOM)
        assert event.source == ""

    def test_source_stored(self):
        event = Event(event_type=EventType.CUSTOM, source="worker-1")
        assert event.source == "worker-1"


class TestEventSerialization:
    """Tests for to_dict / to_json / from_dict / from_json."""

    def test_to_dict_keys(self):
        event = Event(event_type=EventType.JOB_SUBMITTED)
        d = event.to_dict()
        assert set(d.keys()) == {"event_type", "payload", "event_id", "timestamp", "source"}

    def test_to_dict_event_type_is_string(self):
        event = Event(event_type=EventType.JOB_SUBMITTED)
        assert event.to_dict()["event_type"] == "job.submitted"

    def test_to_json_returns_valid_json(self):
        event = Event(event_type=EventType.JOB_COMPLETED, payload={"pages": 5})
        parsed = json.loads(event.to_json())
        assert parsed["event_type"] == "job.completed"
        assert parsed["payload"]["pages"] == 5

    def test_from_dict_roundtrip(self):
        original = Event(
            event_type=EventType.PAGE_PROCESSED,
            payload={"page": 3},
            source="worker-2",
        )
        restored = Event.from_dict(original.to_dict())
        assert restored.event_type == original.event_type
        assert restored.payload == original.payload
        assert restored.event_id == original.event_id
        assert restored.timestamp == original.timestamp
        assert restored.source == original.source

    def test_from_json_roundtrip(self):
        original = Event(
            event_type=EventType.ALERT_TRIGGERED,
            payload={"alert": "queue_full"},
        )
        restored = Event.from_json(original.to_json())
        assert restored.event_type == original.event_type
        assert restored.payload == original.payload

    def test_from_dict_unknown_event_type_falls_back_to_custom(self):
        data = {
            "event_type": "totally.unknown",
            "payload": {},
            "event_id": "abc",
            "timestamp": 1.0,
            "source": "",
        }
        event = Event.from_dict(data)
        assert event.event_type == EventType.CUSTOM

    def test_from_dict_missing_event_type_defaults_to_custom(self):
        event = Event.from_dict({"payload": {"x": 1}})
        assert event.event_type == EventType.CUSTOM

    def test_from_dict_preserves_explicit_fields(self):
        data = {
            "event_type": "job.failed",
            "payload": {"code": 500},
            "event_id": "eid-999",
            "timestamp": 42.0,
            "source": "gpu-3",
        }
        event = Event.from_dict(data)
        assert event.event_id == "eid-999"
        assert event.timestamp == 42.0
        assert event.source == "gpu-3"


# ---------------------------------------------------------------------------
# EventBus ABC
# ---------------------------------------------------------------------------


class TestEventBusABC:
    """Tests for the abstract EventBus class."""

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            EventBus()


# ---------------------------------------------------------------------------
# LocalEventBus
# ---------------------------------------------------------------------------


class TestLocalEventBusPublish:
    """Tests for LocalEventBus publish and callback dispatch."""

    def test_publish_returns_true(self):
        bus = LocalEventBus()
        event = Event(event_type=EventType.JOB_SUBMITTED)
        assert bus.publish(event) is True

    def test_publish_fires_matching_callback(self):
        bus = LocalEventBus()
        received = []
        bus.subscribe(EventType.JOB_SUBMITTED, lambda e: received.append(e))
        event = Event(event_type=EventType.JOB_SUBMITTED, payload={"job": "1"})
        bus.publish(event)
        assert len(received) == 1
        assert received[0].payload == {"job": "1"}

    def test_publish_does_not_fire_non_matching_callback(self):
        bus = LocalEventBus()
        received = []
        bus.subscribe(EventType.JOB_FAILED, lambda e: received.append(e))
        bus.publish(Event(event_type=EventType.JOB_SUBMITTED))
        assert len(received) == 0

    def test_publish_multiple_subscribers(self):
        bus = LocalEventBus()
        r1, r2 = [], []
        bus.subscribe(EventType.JOB_COMPLETED, lambda e: r1.append(e))
        bus.subscribe(EventType.JOB_COMPLETED, lambda e: r2.append(e))
        bus.publish(Event(event_type=EventType.JOB_COMPLETED))
        assert len(r1) == 1
        assert len(r2) == 1

    def test_callback_exception_is_logged_not_raised(self):
        bus = LocalEventBus()

        def bad_callback(e):
            raise RuntimeError("boom")

        bus.subscribe(EventType.JOB_SUBMITTED, bad_callback)
        # Should not raise
        result = bus.publish(Event(event_type=EventType.JOB_SUBMITTED))
        assert result is True

    def test_callback_error_does_not_block_other_subscribers(self):
        bus = LocalEventBus()
        received = []

        def bad(e):
            raise ValueError("fail")

        bus.subscribe(EventType.JOB_SUBMITTED, bad)
        bus.subscribe(EventType.JOB_SUBMITTED, lambda e: received.append(e))
        bus.publish(Event(event_type=EventType.JOB_SUBMITTED))
        assert len(received) == 1


class TestLocalEventBusWildcard:
    """Tests for CUSTOM type acting as wildcard subscription."""

    def test_custom_type_receives_all_events(self):
        bus = LocalEventBus()
        received = []
        bus.subscribe(EventType.CUSTOM, lambda e: received.append(e))
        bus.publish(Event(event_type=EventType.JOB_SUBMITTED))
        bus.publish(Event(event_type=EventType.JOB_COMPLETED))
        bus.publish(Event(event_type=EventType.PAGE_PROCESSED))
        assert len(received) == 3


class TestLocalEventBusSubscribe:
    """Tests for subscribe / unsubscribe."""

    def test_subscribe_returns_id_string(self):
        bus = LocalEventBus()
        sub_id = bus.subscribe(EventType.JOB_STARTED, lambda e: None)
        assert isinstance(sub_id, str)
        assert sub_id.startswith("local-sub-")

    def test_subscribe_returns_unique_ids(self):
        bus = LocalEventBus()
        ids = {bus.subscribe(EventType.JOB_STARTED, lambda e: None) for _ in range(10)}
        assert len(ids) == 10

    def test_unsubscribe_returns_true(self):
        bus = LocalEventBus()
        sub_id = bus.subscribe(EventType.JOB_STARTED, lambda e: None)
        assert bus.unsubscribe(sub_id) is True

    def test_unsubscribe_unknown_returns_false(self):
        bus = LocalEventBus()
        assert bus.unsubscribe("nonexistent-sub-999") is False

    def test_unsubscribe_stops_callback(self):
        bus = LocalEventBus()
        received = []
        sub_id = bus.subscribe(EventType.JOB_STARTED, lambda e: received.append(e))
        bus.publish(Event(event_type=EventType.JOB_STARTED))
        assert len(received) == 1
        bus.unsubscribe(sub_id)
        bus.publish(Event(event_type=EventType.JOB_STARTED))
        assert len(received) == 1  # no new events


class TestLocalEventBusHistory:
    """Tests for get_history and clear_history."""

    def test_get_history_returns_published_events(self):
        bus = LocalEventBus()
        bus.publish(Event(event_type=EventType.JOB_SUBMITTED))
        bus.publish(Event(event_type=EventType.JOB_COMPLETED))
        history = bus.get_history()
        assert len(history) == 2

    def test_get_history_filter_by_type(self):
        bus = LocalEventBus()
        bus.publish(Event(event_type=EventType.JOB_SUBMITTED))
        bus.publish(Event(event_type=EventType.JOB_COMPLETED))
        bus.publish(Event(event_type=EventType.JOB_SUBMITTED))
        history = bus.get_history(event_type=EventType.JOB_SUBMITTED)
        assert len(history) == 2

    def test_get_history_with_limit(self):
        bus = LocalEventBus()
        for _ in range(20):
            bus.publish(Event(event_type=EventType.JOB_SUBMITTED))
        history = bus.get_history(limit=5)
        assert len(history) == 5

    def test_get_history_limit_returns_most_recent(self):
        bus = LocalEventBus()
        for i in range(10):
            bus.publish(Event(event_type=EventType.JOB_SUBMITTED, payload={"i": i}))
        history = bus.get_history(limit=3)
        assert [e.payload["i"] for e in history] == [7, 8, 9]

    def test_clear_history(self):
        bus = LocalEventBus()
        bus.publish(Event(event_type=EventType.JOB_SUBMITTED))
        bus.clear_history()
        assert bus.get_history() == []

    def test_history_eviction_at_max(self):
        bus = LocalEventBus()
        bus._max_history = 5
        for i in range(10):
            bus.publish(Event(event_type=EventType.JOB_SUBMITTED, payload={"i": i}))
        history = bus.get_history()
        assert len(history) == 5
        assert history[0].payload["i"] == 5


class TestLocalEventBusThreadSafety:
    """Tests for concurrent publish/subscribe operations."""

    def test_concurrent_publish(self):
        bus = LocalEventBus()
        count = 200
        barrier = threading.Barrier(4)

        def publisher():
            barrier.wait()
            for _ in range(count):
                bus.publish(Event(event_type=EventType.JOB_SUBMITTED))

        threads = [threading.Thread(target=publisher) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(bus.get_history(limit=count * 4)) == count * 4

    def test_concurrent_subscribe_unsubscribe(self):
        bus = LocalEventBus()
        ids = []
        lock = threading.Lock()

        def subscriber():
            sub_id = bus.subscribe(EventType.JOB_COMPLETED, lambda e: None)
            with lock:
                ids.append(sub_id)

        threads = [threading.Thread(target=subscriber) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(ids) == 20
        for sub_id in ids:
            assert bus.unsubscribe(sub_id) is True


# ---------------------------------------------------------------------------
# KafkaEventBus
# ---------------------------------------------------------------------------


class TestKafkaEventBus:
    """Tests for KafkaEventBus (no real Kafka required)."""

    def test_init_stores_config(self):
        bus = KafkaEventBus(bootstrap_servers="kafka:9093", topic_prefix="myapp")
        assert bus._bootstrap_servers == "kafka:9093"
        assert bus._topic_prefix == "myapp"

    def test_topic_for_job_submitted_legacy(self):
        bus = KafkaEventBus(topic_prefix="ocr", use_topic_map=False)
        assert bus._topic_for(EventType.JOB_SUBMITTED) == "ocr.job.submitted"

    def test_topic_for_page_processed_legacy(self):
        bus = KafkaEventBus(topic_prefix="pipeline", use_topic_map=False)
        assert bus._topic_for(EventType.PAGE_PROCESSED) == "pipeline.page.processed"

    def test_topic_for_job_submitted_with_map(self):
        bus = KafkaEventBus(use_topic_map=True)
        assert bus._topic_for(EventType.JOB_SUBMITTED) == "ocr.jobs.submitted"

    def test_topic_for_page_processed_with_map(self):
        bus = KafkaEventBus(use_topic_map=True)
        assert bus._topic_for(EventType.PAGE_PROCESSED) == "ocr.pages.processed"

    def test_get_producer_raises_import_error_when_missing(self):
        bus = KafkaEventBus()
        with patch.dict(sys.modules, {"kafka": None}):
            with pytest.raises(ImportError, match="kafka-python is required"):
                bus._get_producer()

    def test_subscribe_returns_kafka_sub_id(self):
        bus = KafkaEventBus()
        sub_id = bus.subscribe(EventType.JOB_STARTED, lambda e: None)
        assert sub_id.startswith("kafka-sub-")

    def test_unsubscribe_returns_true_for_known(self):
        bus = KafkaEventBus()
        sub_id = bus.subscribe(EventType.JOB_STARTED, lambda e: None)
        assert bus.unsubscribe(sub_id) is True

    def test_unsubscribe_returns_false_for_unknown(self):
        bus = KafkaEventBus()
        assert bus.unsubscribe("kafka-sub-999") is False

    def test_publish_returns_false_when_kafka_unavailable(self):
        bus = KafkaEventBus()
        with patch.dict(sys.modules, {"kafka": None}):
            result = bus.publish(Event(event_type=EventType.JOB_SUBMITTED))
        assert result is False


# ---------------------------------------------------------------------------
# SnsEventBus
# ---------------------------------------------------------------------------


class TestSnsEventBus:
    """Tests for SnsEventBus (no real AWS required)."""

    def test_init_stores_config(self):
        bus = SnsEventBus(region="eu-west-1", topic_arn_prefix="arn:aws:sns:eu-west-1:123")
        assert bus._region == "eu-west-1"
        assert bus._topic_arn_prefix == "arn:aws:sns:eu-west-1:123"

    def test_get_client_raises_import_error_when_missing(self):
        bus = SnsEventBus()
        with patch.dict(sys.modules, {"boto3": None}):
            with pytest.raises(ImportError, match="boto3 is required"):
                bus._get_client()

    def test_subscribe_returns_sns_sub_id(self):
        bus = SnsEventBus()
        sub_id = bus.subscribe(EventType.ALERT_TRIGGERED, lambda e: None)
        assert sub_id.startswith("sns-sub-")

    def test_unsubscribe_returns_true_for_known(self):
        bus = SnsEventBus()
        sub_id = bus.subscribe(EventType.ALERT_TRIGGERED, lambda e: None)
        assert bus.unsubscribe(sub_id) is True

    def test_unsubscribe_returns_false_for_unknown(self):
        bus = SnsEventBus()
        assert bus.unsubscribe("sns-sub-999") is False

    def test_publish_returns_false_when_boto3_unavailable(self):
        bus = SnsEventBus()
        with patch.dict(sys.modules, {"boto3": None}):
            result = bus.publish(Event(event_type=EventType.JOB_COMPLETED))
        assert result is False


# ---------------------------------------------------------------------------
# create_event_bus factory
# ---------------------------------------------------------------------------


class TestCreateEventBus:
    """Tests for the create_event_bus factory function."""

    def test_create_local(self):
        bus = create_event_bus("local")
        assert isinstance(bus, LocalEventBus)

    def test_create_kafka(self):
        bus = create_event_bus("kafka", bootstrap_servers="kafka:9092")
        assert isinstance(bus, KafkaEventBus)
        assert bus._bootstrap_servers == "kafka:9092"

    def test_create_sns(self):
        bus = create_event_bus("sns", region="us-west-2")
        assert isinstance(bus, SnsEventBus)
        assert bus._region == "us-west-2"

    def test_create_unsupported_raises(self):
        with pytest.raises(ValueError, match="Unsupported event bus backend"):
            create_event_bus("rabbitmq")

    def test_create_default_is_local(self):
        bus = create_event_bus()
        assert isinstance(bus, LocalEventBus)
