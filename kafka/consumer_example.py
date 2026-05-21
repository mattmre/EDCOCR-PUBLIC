#!/usr/bin/env python3
"""Example Kafka consumer for EDCOCR event bus.

Demonstrates consuming events from the ocr.jobs.completed topic with:
  - Automatic deserialization to Event objects
  - Graceful error handling and shutdown
  - Both single-message and batch consumption patterns
  - Fallback between confluent-kafka and kafka-python

Usage:
    # Single-message consumption (default)
    python kafka/consumer_example.py

    # Batch consumption mode
    python kafka/consumer_example.py --batch --batch-size 10

    # Custom bootstrap server and topic
    python kafka/consumer_example.py --bootstrap-servers kafka:9092 --topic ocr.jobs.failed

    # With consumer group
    python kafka/consumer_example.py --group-id my-downstream-service

Environment variables:
    KAFKA_BOOTSTRAP_SERVERS — broker address (default: localhost:9094)
    KAFKA_CONSUMER_GROUP   — consumer group ID (default: ocr-example-consumer)
    KAFKA_TOPIC            — topic to consume (default: ocr.jobs.completed)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time

# Add project root to path so we can import api.event_bus
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from api.event_bus import Event, EventType  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("kafka-consumer-example")

# Graceful shutdown flag
_SHUTDOWN = False


def _signal_handler(signum, frame):
    """Set shutdown flag on SIGINT/SIGTERM."""
    global _SHUTDOWN
    logger.info("Shutdown signal received (signal=%d)", signum)
    _SHUTDOWN = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ---------------------------------------------------------------------------
# Consumer backend abstraction
# ---------------------------------------------------------------------------


def _create_consumer_confluent(bootstrap_servers: str, group_id: str, topic: str):
    """Create a consumer using confluent-kafka (preferred for production)."""
    from confluent_kafka import Consumer as ConfluentConsumer

    conf = {
        "bootstrap.servers": bootstrap_servers,
        "group.id": group_id,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": True,
        "auto.commit.interval.ms": 5000,
        "session.timeout.ms": 30000,
        "max.poll.interval.ms": 300000,
    }
    consumer = ConfluentConsumer(conf)
    consumer.subscribe([topic])
    logger.info(
        "Created confluent-kafka consumer (servers=%s, group=%s, topic=%s)",
        bootstrap_servers,
        group_id,
        topic,
    )
    return ("confluent", consumer)


def _create_consumer_kafka_python(bootstrap_servers: str, group_id: str, topic: str):
    """Create a consumer using kafka-python (fallback)."""
    from kafka import KafkaConsumer

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers.split(","),
        group_id=group_id,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        auto_commit_interval_ms=5000,
        value_deserializer=lambda m: m.decode("utf-8") if m else None,
        consumer_timeout_ms=1000,
    )
    logger.info(
        "Created kafka-python consumer (servers=%s, group=%s, topic=%s)",
        bootstrap_servers,
        group_id,
        topic,
    )
    return ("kafka-python", consumer)


def create_consumer(bootstrap_servers: str, group_id: str, topic: str):
    """Create a Kafka consumer with automatic backend selection.

    Tries confluent-kafka first, then falls back to kafka-python.
    Raises ImportError if neither is available.
    """
    try:
        return _create_consumer_confluent(bootstrap_servers, group_id, topic)
    except ImportError:
        logger.info("confluent-kafka not available, trying kafka-python")

    try:
        return _create_consumer_kafka_python(bootstrap_servers, group_id, topic)
    except ImportError:
        raise ImportError(
            "Neither confluent-kafka nor kafka-python is installed. "
            "Install one with: pip install confluent-kafka  OR  pip install kafka-python"
        )


# ---------------------------------------------------------------------------
# Message processing
# ---------------------------------------------------------------------------


def process_event(event: Event) -> None:
    """Process a single OCR pipeline event.

    Replace this function body with your downstream integration logic.
    Examples: update a search index, trigger notifications, write to a
    data warehouse, etc.
    """
    logger.info(
        "Received event: type=%s id=%s source=%s",
        event.event_type.value,
        event.event_id,
        event.source,
    )

    if event.event_type == EventType.JOB_COMPLETED:
        job_id = event.payload.get("job_id", "unknown")
        pages = event.payload.get("total_pages", "?")
        logger.info("  Job completed: job_id=%s, pages=%s", job_id, pages)

    elif event.event_type == EventType.JOB_FAILED:
        job_id = event.payload.get("job_id", "unknown")
        error = event.payload.get("error", "unknown")
        logger.warning("  Job failed: job_id=%s, error=%s", job_id, error)

    elif event.event_type == EventType.PAGE_PROCESSED:
        job_id = event.payload.get("job_id", "unknown")
        page = event.payload.get("page_number", "?")
        confidence = event.payload.get("confidence", 0.0)
        logger.info(
            "  Page processed: job_id=%s, page=%s, confidence=%.2f",
            job_id,
            page,
            confidence,
        )

    else:
        logger.info("  Payload: %s", json.dumps(event.payload, default=str)[:200])


def deserialize_message(raw_value) -> Event | None:
    """Deserialize a Kafka message value to an Event object.

    Returns None if deserialization fails (logged, not raised).
    """
    try:
        if isinstance(raw_value, bytes):
            raw_value = raw_value.decode("utf-8")
        if raw_value is None:
            return None
        data = json.loads(raw_value)
        return Event.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("Failed to deserialize message: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Consumption loops
# ---------------------------------------------------------------------------


def consume_single(consumer_backend, consumer) -> None:
    """Single-message consumption loop."""
    logger.info("Starting single-message consumption loop (Ctrl+C to stop)")

    while not _SHUTDOWN:
        if consumer_backend == "confluent":
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                logger.error("Consumer error: %s", msg.error())
                continue
            event = deserialize_message(msg.value())
        else:
            # kafka-python: iteration with consumer_timeout_ms
            try:
                for msg in consumer:
                    if _SHUTDOWN:
                        break
                    event = deserialize_message(msg.value)
                    if event:
                        process_event(event)
                continue
            except StopIteration:
                time.sleep(0.1)
                continue

        if event:
            process_event(event)


def consume_batch(consumer_backend, consumer, batch_size: int = 10) -> None:
    """Batch consumption loop — collects up to batch_size messages."""
    logger.info(
        "Starting batch consumption loop (batch_size=%d, Ctrl+C to stop)",
        batch_size,
    )

    while not _SHUTDOWN:
        batch: list[Event] = []

        if consumer_backend == "confluent":
            messages = consumer.consume(num_messages=batch_size, timeout=2.0)
            for msg in messages:
                if msg.error():
                    logger.error("Consumer error: %s", msg.error())
                    continue
                event = deserialize_message(msg.value())
                if event:
                    batch.append(event)
        else:
            # kafka-python: collect up to batch_size
            deadline = time.time() + 2.0
            try:
                for msg in consumer:
                    if _SHUTDOWN:
                        break
                    event = deserialize_message(msg.value)
                    if event:
                        batch.append(event)
                    if len(batch) >= batch_size or time.time() > deadline:
                        break
            except StopIteration:
                pass

        if batch:
            logger.info("Processing batch of %d events", len(batch))
            for event in batch:
                process_event(event)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="EDCOCR Kafka consumer example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--bootstrap-servers",
        default=os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9094"),
        help="Kafka bootstrap servers (default: localhost:9094)",
    )
    parser.add_argument(
        "--topic",
        default=os.environ.get("KAFKA_TOPIC", "ocr.jobs.completed"),
        help="Topic to consume (default: ocr.jobs.completed)",
    )
    parser.add_argument(
        "--group-id",
        default=os.environ.get("KAFKA_CONSUMER_GROUP", "ocr-example-consumer"),
        help="Consumer group ID (default: ocr-example-consumer)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Enable batch consumption mode",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of messages per batch (default: 10)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    logger.info("EDCOCR Kafka Consumer Example")
    logger.info("Bootstrap servers: %s", args.bootstrap_servers)
    logger.info("Topic: %s", args.topic)
    logger.info("Group ID: %s", args.group_id)
    logger.info("Mode: %s", "batch" if args.batch else "single")

    try:
        backend, consumer = create_consumer(
            args.bootstrap_servers, args.group_id, args.topic
        )
    except ImportError as exc:
        logger.error(str(exc))
        return 1
    except Exception as exc:
        logger.error("Failed to create consumer: %s", exc)
        return 1

    try:
        if args.batch:
            consume_batch(backend, consumer, args.batch_size)
        else:
            consume_single(backend, consumer)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        logger.info("Closing consumer...")
        if backend == "confluent":
            consumer.close()
        else:
            consumer.close()

    logger.info("Consumer stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
