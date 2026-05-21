"""Event-driven integrations for Kafka, SNS/SQS, and local dispatch.

Provides a unified EventBus interface for publishing and subscribing
to OCR pipeline events (job.submitted, job.completed, job.failed, etc.)
across different messaging backends.

All external SDK imports are lazy — module works without dependencies.

Environment variables:
    EVENT_BUS_BACKEND           — "local" (default), "kafka", or "sns"
    KAFKA_BOOTSTRAP_SERVERS     — Kafka broker address (default: localhost:9092)
    KAFKA_TOPIC_PREFIX          — Prefix for Kafka topic names (default: ocr-pipeline)
    KAFKA_SECURITY_PROTOCOL     — Security protocol (PLAINTEXT, SASL_SSL, etc.)
    KAFKA_SASL_MECHANISM        — SASL mechanism (PLAIN, SCRAM-SHA-256, AWS_MSK_IAM)
    KAFKA_SASL_USERNAME         — SASL username (for Confluent Cloud / SCRAM)
    KAFKA_SASL_PASSWORD         — SASL password (for Confluent Cloud / SCRAM)
    SNS_REGION                  — AWS region for SNS backend
    SNS_TOPIC_ARN_PREFIX        — ARN prefix for SNS topics
"""

import json
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Topic name mapping: EventType -> canonical Kafka topic name
# Overridable via KAFKA_TOPIC_<EVENTTYPE> env vars (e.g. KAFKA_TOPIC_JOB_SUBMITTED)
# ---------------------------------------------------------------------------

DEFAULT_TOPIC_MAP: dict[str, str] = {
    "job.submitted": "ocr.jobs.submitted",
    "job.started": "ocr.jobs.submitted",
    "job.completed": "ocr.jobs.completed",
    "job.failed": "ocr.jobs.failed",
    "job.cancelled": "ocr.jobs.failed",
    "page.processed": "ocr.pages.processed",
    "worker.online": "ocr.jobs.submitted",
    "worker.offline": "ocr.jobs.submitted",
    "alert.triggered": "ocr.jobs.submitted",
    "custom": "ocr.jobs.submitted",
}


def get_topic_for_event(event_type_value: str, prefix: str = "") -> str:
    """Resolve the Kafka topic name for a given event type value.

    Checks for an env var override first (e.g. KAFKA_TOPIC_JOB_SUBMITTED),
    then falls back to DEFAULT_TOPIC_MAP, then to prefix-based derivation.
    """
    # Check env var override: KAFKA_TOPIC_JOB_SUBMITTED, KAFKA_TOPIC_PAGE_PROCESSED, etc.
    env_key = "KAFKA_TOPIC_" + event_type_value.upper().replace(".", "_")
    env_override = os.environ.get(env_key)
    if env_override:
        return env_override

    # Check default topic map
    if event_type_value in DEFAULT_TOPIC_MAP:
        topic = DEFAULT_TOPIC_MAP[event_type_value]
        if prefix:
            return f"{prefix}.{topic}"
        return topic

    # Fallback: derive from prefix + event type
    if prefix:
        return f"{prefix}.{event_type_value}"
    return f"ocr-pipeline.{event_type_value}"


class EventType(Enum):
    JOB_SUBMITTED = "job.submitted"
    JOB_STARTED = "job.started"
    JOB_COMPLETED = "job.completed"
    JOB_FAILED = "job.failed"
    JOB_CANCELLED = "job.cancelled"
    PAGE_PROCESSED = "page.processed"
    WORKER_ONLINE = "worker.online"
    WORKER_OFFLINE = "worker.offline"
    ALERT_TRIGGERED = "alert.triggered"
    CUSTOM = "custom"


@dataclass
class Event:
    """A pipeline event."""

    event_type: EventType
    payload: dict = field(default_factory=dict)
    event_id: str = ""
    timestamp: float = 0.0
    source: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()
        if not self.event_id:
            import uuid

            self.event_id = str(uuid.uuid4())

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type.value,
            "payload": self.payload,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "source": self.source,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict) -> "Event":
        et = data.get("event_type", "custom")
        try:
            event_type = EventType(et)
        except ValueError:
            event_type = EventType.CUSTOM
        return cls(
            event_type=event_type,
            payload=data.get("payload", {}),
            event_id=data.get("event_id", ""),
            timestamp=data.get("timestamp", 0.0),
            source=data.get("source", ""),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "Event":
        return cls.from_dict(json.loads(json_str))


class EventBus(ABC):
    """Abstract event bus interface."""

    @abstractmethod
    def publish(self, event: Event) -> bool:
        """Publish an event. Returns True if successful."""
        ...

    @abstractmethod
    def subscribe(self, event_type: EventType, callback) -> str:
        """Subscribe to an event type. Returns subscription ID."""
        ...

    @abstractmethod
    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe. Returns True if found and removed."""
        ...


class LocalEventBus(EventBus):
    """In-process event bus using threading callbacks."""

    def __init__(self):
        self._lock = threading.Lock()
        self._subscriptions: dict = {}  # sub_id -> (EventType, callback)
        self._counter = 0
        self._history: list = []  # List of Event for debugging
        self._max_history = 1000

    def publish(self, event: Event) -> bool:
        with self._lock:
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]
            subs = list(self._subscriptions.items())

        for sub_id, (et, callback) in subs:
            if et == event.event_type or et == EventType.CUSTOM:
                try:
                    callback(event)
                except Exception:
                    logger.exception(f"Event callback failed for {sub_id}")
        return True

    def subscribe(self, event_type: EventType, callback) -> str:
        with self._lock:
            self._counter += 1
            sub_id = f"local-sub-{self._counter}"
            self._subscriptions[sub_id] = (event_type, callback)
        return sub_id

    def unsubscribe(self, subscription_id: str) -> bool:
        with self._lock:
            return self._subscriptions.pop(subscription_id, None) is not None

    def get_history(self, event_type: EventType = None, limit: int = 100) -> list:
        with self._lock:
            if event_type:
                filtered = [e for e in self._history if e.event_type == event_type]
            else:
                filtered = list(self._history)
        return filtered[-limit:]

    def clear_history(self):
        with self._lock:
            self._history.clear()


class KafkaEventBus(EventBus):
    """Apache Kafka event bus. Requires kafka-python package.

    Configuration can be provided via constructor args or environment
    variables (constructor args take precedence):

        KAFKA_BOOTSTRAP_SERVERS — broker address (default: localhost:9092)
        KAFKA_TOPIC_PREFIX      — topic name prefix (default: ocr-pipeline)
        KAFKA_SECURITY_PROTOCOL — PLAINTEXT, SASL_SSL, etc.
        KAFKA_SASL_MECHANISM    — PLAIN, SCRAM-SHA-256, AWS_MSK_IAM
        KAFKA_SASL_USERNAME     — SASL username
        KAFKA_SASL_PASSWORD     — SASL password

    Topic names are resolved via get_topic_for_event() which checks
    KAFKA_TOPIC_<EVENT_TYPE> env vars, then DEFAULT_TOPIC_MAP, then
    falls back to prefix-based derivation.

    When use_topic_map is True (default), published messages use the
    canonical topic names from DEFAULT_TOPIC_MAP (e.g. ocr.jobs.completed).
    When False, topics use the legacy prefix.event_type format.
    """

    def __init__(
        self,
        bootstrap_servers: str = "",
        topic_prefix: str = "",
        use_topic_map: bool = True,
        security_protocol: str = "",
        sasl_mechanism: str = "",
        sasl_username: str = "",
        sasl_password: str = "",
    ):
        self._bootstrap_servers = (
            bootstrap_servers
            or os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        )
        self._topic_prefix = (
            topic_prefix
            or os.environ.get("KAFKA_TOPIC_PREFIX", "ocr-pipeline")
        )
        self._use_topic_map = use_topic_map
        self._security_protocol = (
            security_protocol
            or os.environ.get("KAFKA_SECURITY_PROTOCOL", "")
        )
        self._sasl_mechanism = (
            sasl_mechanism
            or os.environ.get("KAFKA_SASL_MECHANISM", "")
        )
        self._sasl_username = (
            sasl_username
            or os.environ.get("KAFKA_SASL_USERNAME", "")
        )
        self._sasl_password = (
            sasl_password
            or os.environ.get("KAFKA_SASL_PASSWORD", "")
        )
        self._producer = None
        self._local_subs: dict = {}
        self._counter = 0
        self._lock = threading.Lock()

    def _get_producer(self):
        if self._producer is None:
            try:
                from kafka import KafkaProducer
            except ImportError:
                raise ImportError(
                    "kafka-python is required. "
                    "Install with: pip install kafka-python"
                )
            producer_kwargs: dict = {
                "bootstrap_servers": self._bootstrap_servers,
                "value_serializer": lambda v: json.dumps(v).encode("utf-8"),
            }
            if self._security_protocol:
                producer_kwargs["security_protocol"] = self._security_protocol
            if self._sasl_mechanism:
                producer_kwargs["sasl_mechanism"] = self._sasl_mechanism
            if self._sasl_username:
                producer_kwargs["sasl_plain_username"] = self._sasl_username
            if self._sasl_password:
                producer_kwargs["sasl_plain_password"] = self._sasl_password
            self._producer = KafkaProducer(**producer_kwargs)
        return self._producer

    def _topic_for(self, event_type: EventType) -> str:
        if self._use_topic_map:
            return get_topic_for_event(event_type.value, prefix="")
        return f"{self._topic_prefix}.{event_type.value}"

    def publish(self, event: Event) -> bool:
        try:
            producer = self._get_producer()
            topic = self._topic_for(event.event_type)
            key = None
            job_id = event.payload.get("job_id")
            if job_id:
                key = str(job_id).encode("utf-8")
            producer.send(topic, value=event.to_dict(), key=key)
            producer.flush()
            return True
        except Exception:
            logger.exception("Kafka publish failed")
            return False

    def subscribe(self, event_type: EventType, callback) -> str:
        with self._lock:
            self._counter += 1
            sub_id = f"kafka-sub-{self._counter}"
            self._local_subs[sub_id] = (event_type, callback)
        return sub_id

    def unsubscribe(self, subscription_id: str) -> bool:
        with self._lock:
            return self._local_subs.pop(subscription_id, None) is not None


class SnsEventBus(EventBus):
    """AWS SNS/SQS event bus. Requires boto3 package."""

    def __init__(self, region: str = "us-east-1", topic_arn_prefix: str = ""):
        self._region = region
        self._topic_arn_prefix = topic_arn_prefix
        self._client = None
        self._local_subs: dict = {}
        self._counter = 0
        self._lock = threading.Lock()

    def _get_client(self):
        if self._client is None:
            try:
                import boto3
            except ImportError:
                raise ImportError(
                    "boto3 is required. Install with: pip install boto3"
                )
            self._client = boto3.client("sns", region_name=self._region)
        return self._client

    def publish(self, event: Event) -> bool:
        try:
            client = self._get_client()
            topic_arn = f"{self._topic_arn_prefix}:{event.event_type.value}"
            client.publish(
                TopicArn=topic_arn,
                Message=event.to_json(),
                Subject=event.event_type.value,
            )
            return True
        except Exception:
            logger.exception("SNS publish failed")
            return False

    def subscribe(self, event_type: EventType, callback) -> str:
        with self._lock:
            self._counter += 1
            sub_id = f"sns-sub-{self._counter}"
            self._local_subs[sub_id] = (event_type, callback)
        return sub_id

    def unsubscribe(self, subscription_id: str) -> bool:
        with self._lock:
            return self._local_subs.pop(subscription_id, None) is not None


def create_event_bus(backend: str = "", **kwargs) -> EventBus:
    """Factory function to create an event bus backend.

    Args:
        backend: "local", "kafka", or "sns".  If empty, reads from
                 EVENT_BUS_BACKEND env var (default: "local").
        **kwargs: Backend-specific configuration.  For Kafka, env vars
                  KAFKA_BOOTSTRAP_SERVERS and KAFKA_TOPIC_PREFIX are
                  used as defaults when not passed explicitly.
    """
    if not backend:
        backend = os.environ.get("EVENT_BUS_BACKEND", "local")

    if backend == "local":
        return LocalEventBus()
    elif backend == "kafka":
        return KafkaEventBus(**kwargs)
    elif backend == "sns":
        return SnsEventBus(**kwargs)
    else:
        raise ValueError(f"Unsupported event bus backend: {backend}")
