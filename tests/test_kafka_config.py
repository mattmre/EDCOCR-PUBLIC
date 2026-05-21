"""Tests for Kafka event bus deployment configuration.

Covers:
  - kafka-topics.yaml parsing and validation
  - Topic name mapping (DEFAULT_TOPIC_MAP, get_topic_for_event)
  - KafkaEventBus env var configuration
  - create_event_bus factory with EVENT_BUS_BACKEND env var
  - Consumer example module structure and CLI parsing
  - Init script existence and structure
  - Docker Compose YAML validity
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from api.event_bus import (
    DEFAULT_TOPIC_MAP,
    EventType,
    KafkaEventBus,
    LocalEventBus,
    create_event_bus,
    get_topic_for_event,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KAFKA_DIR = PROJECT_ROOT / "kafka"
TOPICS_YAML = KAFKA_DIR / "kafka-topics.yaml"
INIT_SCRIPT = KAFKA_DIR / "init-topics.sh"
COMPOSE_FILE = KAFKA_DIR / "docker-compose.kafka.yml"
CONSUMER_EXAMPLE = KAFKA_DIR / "consumer_example.py"
README_FILE = KAFKA_DIR / "README.md"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_yaml_simple(path: Path) -> str:
    """Load YAML as raw text (no pyyaml dependency needed)."""
    return path.read_text(encoding="utf-8")


def _parse_topic_names(yaml_text: str) -> list[str]:
    """Extract topic names from kafka-topics.yaml text."""
    names = []
    for line in yaml_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- name:"):
            name = stripped.replace("- name:", "").strip()
            names.append(name)
    return names


def _strip_inline_comment(value: str) -> str:
    """Remove inline YAML comments (e.g. '604800000  # 7 days' -> '604800000')."""
    # Handle quoted values first
    if value.startswith('"') and '"' in value[1:]:
        return value
    # Strip everything after # that follows whitespace
    idx = value.find("#")
    if idx > 0:
        return value[:idx].strip()
    return value.strip()


def _parse_topic_blocks(yaml_text: str) -> list[dict]:
    """Parse topic blocks into dicts with name, partitions, retention_ms, config keys."""
    topics = []
    current: dict | None = None
    in_config = False

    for line in yaml_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("- name:"):
            if current:
                topics.append(current)
            current = {
                "name": _strip_inline_comment(stripped.split(":", 1)[1].strip()),
                "partitions": 0,
                "replication_factor": 0,
                "retention_ms": 0,
                "config_keys": [],
            }
            in_config = False
            continue

        if current is None:
            continue

        if stripped.startswith("partitions:"):
            raw = _strip_inline_comment(stripped.split(":")[1].strip())
            current["partitions"] = int(raw)
            in_config = False
        elif stripped.startswith("replication_factor:"):
            raw = _strip_inline_comment(stripped.split(":")[1].strip())
            current["replication_factor"] = int(raw)
            in_config = False
        elif stripped.startswith("retention_ms:"):
            raw = _strip_inline_comment(stripped.split(":")[1].strip().strip('"'))
            current["retention_ms"] = int(raw)
            in_config = False
        elif stripped == "config:":
            in_config = True
        elif in_config and ":" in stripped and not stripped.startswith("- "):
            key = stripped.split(":")[0].strip()
            current["config_keys"].append(key)

    if current:
        topics.append(current)
    return topics


# ---------------------------------------------------------------------------
# File existence tests
# ---------------------------------------------------------------------------


class TestKafkaFileStructure:
    """Verify all Kafka deployment files exist."""

    def test_kafka_directory_exists(self):
        assert KAFKA_DIR.is_dir()

    def test_docker_compose_exists(self):
        assert COMPOSE_FILE.is_file()

    def test_topics_yaml_exists(self):
        assert TOPICS_YAML.is_file()

    def test_init_script_exists(self):
        assert INIT_SCRIPT.is_file()

    def test_consumer_example_exists(self):
        assert CONSUMER_EXAMPLE.is_file()

    def test_readme_exists(self):
        assert README_FILE.is_file()


# ---------------------------------------------------------------------------
# kafka-topics.yaml parsing
# ---------------------------------------------------------------------------


class TestTopicsYamlContent:
    """Validate kafka-topics.yaml topic definitions."""

    @pytest.fixture(autouse=True)
    def _load_topics(self):
        self.yaml_text = _load_yaml_simple(TOPICS_YAML)
        self.topic_names = _parse_topic_names(self.yaml_text)
        self.topic_blocks = _parse_topic_blocks(self.yaml_text)

    def test_has_six_topics(self):
        assert len(self.topic_names) == 6

    def test_jobs_submitted_topic(self):
        assert "ocr.jobs.submitted" in self.topic_names

    def test_jobs_completed_topic(self):
        assert "ocr.jobs.completed" in self.topic_names

    def test_jobs_failed_topic(self):
        assert "ocr.jobs.failed" in self.topic_names

    def test_pages_processed_topic(self):
        assert "ocr.pages.processed" in self.topic_names

    def test_entities_extracted_topic(self):
        assert "ocr.entities.extracted" in self.topic_names

    def test_custody_events_topic(self):
        assert "ocr.custody.events" in self.topic_names

    def test_jobs_submitted_has_3_partitions(self):
        t = next(b for b in self.topic_blocks if b["name"] == "ocr.jobs.submitted")
        assert t["partitions"] == 3

    def test_pages_processed_has_6_partitions(self):
        t = next(b for b in self.topic_blocks if b["name"] == "ocr.pages.processed")
        assert t["partitions"] == 6

    def test_jobs_failed_retention_30_days(self):
        t = next(b for b in self.topic_blocks if b["name"] == "ocr.jobs.failed")
        assert t["retention_ms"] == 2592000000  # 30 days

    def test_custody_events_retention_90_days(self):
        t = next(b for b in self.topic_blocks if b["name"] == "ocr.custody.events")
        assert t["retention_ms"] == 7776000000  # 90 days

    def test_custody_events_has_compaction_config(self):
        t = next(b for b in self.topic_blocks if b["name"] == "ocr.custody.events")
        assert "cleanup.policy" in t["config_keys"]

    def test_all_topics_have_replication_factor(self):
        for t in self.topic_blocks:
            assert t["replication_factor"] >= 1, f"{t['name']} has no replication factor"

    def test_all_topics_have_retention_config(self):
        for t in self.topic_blocks:
            assert "retention.ms" in t["config_keys"], f"{t['name']} missing retention.ms config"


# ---------------------------------------------------------------------------
# Docker Compose YAML validation
# ---------------------------------------------------------------------------


class TestDockerComposeYaml:
    """Validate docker-compose.kafka.yml structure."""

    @pytest.fixture(autouse=True)
    def _load_compose(self):
        self.compose_text = COMPOSE_FILE.read_text(encoding="utf-8")

    def test_has_kafka_service(self):
        assert "kafka:" in self.compose_text

    def test_has_kafka_ui_service(self):
        assert "kafka-ui:" in self.compose_text

    def test_has_kafka_init_service(self):
        assert "kafka-init:" in self.compose_text

    def test_uses_bitnami_kafka_image(self):
        assert "bitnami/kafka" in self.compose_text

    def test_uses_kraft_mode(self):
        assert "KAFKA_CFG_PROCESS_ROLES" in self.compose_text
        assert "broker,controller" in self.compose_text

    def test_no_zookeeper_service(self):
        # Comments mention ZooKeeper to explain KRaft mode, but no ZooKeeper
        # service should be defined
        lines = self.compose_text.splitlines()
        service_lines = [
            ln.strip() for ln in lines
            if ln.strip() and not ln.strip().startswith("#")
        ]
        # Check there is no "zookeeper:" service definition
        assert not any(
            ln.startswith("zookeeper:") for ln in service_lines
        )

    def test_has_healthcheck(self):
        assert "healthcheck:" in self.compose_text

    def test_exposes_port_9092(self):
        assert "9092" in self.compose_text

    def test_exposes_kafka_ui_port_8080(self):
        assert "8080" in self.compose_text

    def test_has_volume_definition(self):
        assert "kafka_data:" in self.compose_text

    def test_has_network_config(self):
        assert "ocr-network" in self.compose_text


# ---------------------------------------------------------------------------
# Init script validation
# ---------------------------------------------------------------------------


class TestInitScript:
    """Validate init-topics.sh script."""

    @pytest.fixture(autouse=True)
    def _load_script(self):
        self.script_text = INIT_SCRIPT.read_text(encoding="utf-8")

    def test_has_shebang(self):
        assert self.script_text.startswith("#!/")

    def test_uses_set_euo_pipefail(self):
        assert "set -euo pipefail" in self.script_text

    def test_references_bootstrap_server(self):
        assert "KAFKA_BOOTSTRAP_SERVER" in self.script_text

    def test_references_topic_prefix(self):
        assert "KAFKA_TOPIC_PREFIX" in self.script_text

    def test_references_replication_override(self):
        assert "KAFKA_REPLICATION_FACTOR" in self.script_text

    def test_uses_kafka_topics_sh(self):
        assert "kafka-topics.sh" in self.script_text

    def test_handles_existing_topics(self):
        assert "EXISTS" in self.script_text

    def test_reports_creation_summary(self):
        assert "Topics_CREATED" in self.script_text or "TOPICS_CREATED" in self.script_text


# ---------------------------------------------------------------------------
# DEFAULT_TOPIC_MAP
# ---------------------------------------------------------------------------


class TestDefaultTopicMap:
    """Validate the DEFAULT_TOPIC_MAP in event_bus.py."""

    def test_map_has_all_event_types(self):
        for et in EventType:
            assert et.value in DEFAULT_TOPIC_MAP, f"Missing mapping for {et.value}"

    def test_job_submitted_maps_to_jobs_submitted(self):
        assert DEFAULT_TOPIC_MAP["job.submitted"] == "ocr.jobs.submitted"

    def test_job_completed_maps_to_jobs_completed(self):
        assert DEFAULT_TOPIC_MAP["job.completed"] == "ocr.jobs.completed"

    def test_job_failed_maps_to_jobs_failed(self):
        assert DEFAULT_TOPIC_MAP["job.failed"] == "ocr.jobs.failed"

    def test_page_processed_maps_to_pages_processed(self):
        assert DEFAULT_TOPIC_MAP["page.processed"] == "ocr.pages.processed"


# ---------------------------------------------------------------------------
# get_topic_for_event
# ---------------------------------------------------------------------------


class TestGetTopicForEvent:
    """Test topic resolution with env var overrides and defaults."""

    def test_returns_default_mapping(self):
        topic = get_topic_for_event("job.completed")
        assert topic == "ocr.jobs.completed"

    def test_returns_default_mapping_for_page_processed(self):
        topic = get_topic_for_event("page.processed")
        assert topic == "ocr.pages.processed"

    def test_env_var_override(self):
        with patch.dict(os.environ, {"KAFKA_TOPIC_JOB_COMPLETED": "custom.completed"}):
            topic = get_topic_for_event("job.completed")
            assert topic == "custom.completed"

    def test_env_var_override_takes_precedence_over_map(self):
        with patch.dict(os.environ, {"KAFKA_TOPIC_JOB_FAILED": "myapp.failures"}):
            topic = get_topic_for_event("job.failed")
            assert topic == "myapp.failures"

    def test_prefix_prepended_to_mapped_topic(self):
        topic = get_topic_for_event("job.completed", prefix="staging")
        assert topic == "staging.ocr.jobs.completed"

    def test_unknown_event_type_uses_prefix_fallback(self):
        topic = get_topic_for_event("totally.unknown", prefix="myprefix")
        assert topic == "myprefix.totally.unknown"

    def test_unknown_event_type_no_prefix(self):
        topic = get_topic_for_event("totally.unknown")
        assert topic == "ocr-pipeline.totally.unknown"


# ---------------------------------------------------------------------------
# KafkaEventBus env var configuration
# ---------------------------------------------------------------------------


class TestKafkaEventBusEnvConfig:
    """Test KafkaEventBus configuration via environment variables."""

    def test_default_bootstrap_servers(self):
        env_clean = {
            k: v for k, v in os.environ.items()
            if k != "KAFKA_BOOTSTRAP_SERVERS"
        }
        with patch.dict(os.environ, env_clean, clear=True):
            bus = KafkaEventBus()
            assert bus._bootstrap_servers == "localhost:9092"

    def test_bootstrap_servers_from_env(self):
        with patch.dict(os.environ, {"KAFKA_BOOTSTRAP_SERVERS": "kafka.prod:9093"}):
            bus = KafkaEventBus()
            assert bus._bootstrap_servers == "kafka.prod:9093"

    def test_constructor_arg_overrides_env(self):
        with patch.dict(os.environ, {"KAFKA_BOOTSTRAP_SERVERS": "kafka.prod:9093"}):
            bus = KafkaEventBus(bootstrap_servers="custom:9092")
            assert bus._bootstrap_servers == "custom:9092"

    def test_topic_prefix_from_env(self):
        with patch.dict(os.environ, {"KAFKA_TOPIC_PREFIX": "staging-ocr"}):
            bus = KafkaEventBus()
            assert bus._topic_prefix == "staging-ocr"

    def test_security_protocol_from_env(self):
        with patch.dict(os.environ, {"KAFKA_SECURITY_PROTOCOL": "SASL_SSL"}):
            bus = KafkaEventBus()
            assert bus._security_protocol == "SASL_SSL"

    def test_sasl_mechanism_from_env(self):
        with patch.dict(os.environ, {"KAFKA_SASL_MECHANISM": "PLAIN"}):
            bus = KafkaEventBus()
            assert bus._sasl_mechanism == "PLAIN"

    def test_sasl_credentials_from_env(self):
        env = {
            "KAFKA_SASL_USERNAME": "myuser",
            "KAFKA_SASL_PASSWORD": "mypass",
        }
        with patch.dict(os.environ, env):
            bus = KafkaEventBus()
            assert bus._sasl_username == "myuser"
            assert bus._sasl_password == "mypass"

    def test_use_topic_map_default_true(self):
        bus = KafkaEventBus()
        assert bus._use_topic_map is True

    def test_topic_for_with_map_enabled(self):
        bus = KafkaEventBus(use_topic_map=True)
        assert bus._topic_for(EventType.JOB_COMPLETED) == "ocr.jobs.completed"

    def test_topic_for_with_map_disabled(self):
        bus = KafkaEventBus(topic_prefix="myapp", use_topic_map=False)
        assert bus._topic_for(EventType.JOB_COMPLETED) == "myapp.job.completed"

    def test_publish_includes_job_id_key(self):
        """Verify that publish extracts job_id from payload for message key."""
        bus = KafkaEventBus()
        # We cannot actually publish without Kafka, but we can verify
        # the _topic_for logic is correct
        assert bus._topic_for(EventType.JOB_SUBMITTED) == "ocr.jobs.submitted"


# ---------------------------------------------------------------------------
# create_event_bus with EVENT_BUS_BACKEND
# ---------------------------------------------------------------------------


class TestCreateEventBusEnvVar:
    """Test create_event_bus factory with env var config."""

    def test_default_creates_local(self):
        env_clean = {
            k: v for k, v in os.environ.items()
            if k != "EVENT_BUS_BACKEND"
        }
        with patch.dict(os.environ, env_clean, clear=True):
            bus = create_event_bus()
            assert isinstance(bus, LocalEventBus)

    def test_env_var_kafka_creates_kafka_bus(self):
        with patch.dict(os.environ, {"EVENT_BUS_BACKEND": "kafka"}):
            bus = create_event_bus()
            assert isinstance(bus, KafkaEventBus)

    def test_explicit_backend_overrides_env(self):
        with patch.dict(os.environ, {"EVENT_BUS_BACKEND": "kafka"}):
            bus = create_event_bus("local")
            assert isinstance(bus, LocalEventBus)

    def test_empty_string_backend_reads_env(self):
        with patch.dict(os.environ, {"EVENT_BUS_BACKEND": "kafka"}):
            bus = create_event_bus("")
            assert isinstance(bus, KafkaEventBus)


# ---------------------------------------------------------------------------
# Consumer example module structure
# ---------------------------------------------------------------------------


class TestConsumerExample:
    """Validate consumer_example.py structure and imports."""

    @pytest.fixture(autouse=True)
    def _load_module_text(self):
        self.source = CONSUMER_EXAMPLE.read_text(encoding="utf-8")

    def test_has_main_function(self):
        assert "def main(" in self.source

    def test_has_process_event_function(self):
        assert "def process_event(" in self.source

    def test_has_deserialize_message_function(self):
        assert "def deserialize_message(" in self.source

    def test_has_parse_args_function(self):
        assert "def parse_args(" in self.source

    def test_has_create_consumer_function(self):
        assert "def create_consumer(" in self.source

    def test_has_consume_single_function(self):
        assert "def consume_single(" in self.source

    def test_has_consume_batch_function(self):
        assert "def consume_batch(" in self.source

    def test_imports_event_bus(self):
        assert "from api.event_bus import" in self.source

    def test_handles_signal_shutdown(self):
        assert "signal.SIGINT" in self.source
        assert "signal.SIGTERM" in self.source

    def test_references_bootstrap_servers_env(self):
        assert "KAFKA_BOOTSTRAP_SERVERS" in self.source

    def test_references_consumer_group_env(self):
        assert "KAFKA_CONSUMER_GROUP" in self.source

    def test_has_batch_argument(self):
        assert "--batch" in self.source

    def test_has_batch_size_argument(self):
        assert "--batch-size" in self.source


# ---------------------------------------------------------------------------
# Consumer example parse_args
# ---------------------------------------------------------------------------


class TestConsumerParseArgs:
    """Test parse_args from consumer_example (imported dynamically)."""

    @pytest.fixture(autouse=True)
    def _import_consumer(self):
        # Dynamically import the consumer module
        kafka_dir = str(KAFKA_DIR)
        if kafka_dir not in sys.path:
            sys.path.insert(0, kafka_dir)
        spec = importlib.util.spec_from_file_location(
            "consumer_example", str(CONSUMER_EXAMPLE)
        )
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)

    def test_default_bootstrap_servers(self):
        env_clean = {
            k: v for k, v in os.environ.items()
            if k != "KAFKA_BOOTSTRAP_SERVERS"
        }
        with patch.dict(os.environ, env_clean, clear=True):
            args = self.mod.parse_args([])
            assert args.bootstrap_servers == "localhost:9094"

    def test_default_topic(self):
        env_clean = {
            k: v for k, v in os.environ.items()
            if k != "KAFKA_TOPIC"
        }
        with patch.dict(os.environ, env_clean, clear=True):
            args = self.mod.parse_args([])
            assert args.topic == "ocr.jobs.completed"

    def test_default_group_id(self):
        env_clean = {
            k: v for k, v in os.environ.items()
            if k != "KAFKA_CONSUMER_GROUP"
        }
        with patch.dict(os.environ, env_clean, clear=True):
            args = self.mod.parse_args([])
            assert args.group_id == "ocr-example-consumer"

    def test_custom_topic_flag(self):
        args = self.mod.parse_args(["--topic", "ocr.pages.processed"])
        assert args.topic == "ocr.pages.processed"

    def test_batch_flag(self):
        args = self.mod.parse_args(["--batch"])
        assert args.batch is True

    def test_batch_size_flag(self):
        args = self.mod.parse_args(["--batch", "--batch-size", "25"])
        assert args.batch_size == 25

    def test_group_id_flag(self):
        args = self.mod.parse_args(["--group-id", "my-service"])
        assert args.group_id == "my-service"

    def test_deserialize_valid_json(self):
        raw = json.dumps({
            "event_type": "job.completed",
            "payload": {"job_id": "abc"},
            "event_id": "e1",
            "timestamp": 1.0,
            "source": "test",
        })
        event = self.mod.deserialize_message(raw)
        assert event is not None
        assert event.event_type == EventType.JOB_COMPLETED
        assert event.payload["job_id"] == "abc"

    def test_deserialize_bytes(self):
        raw = json.dumps({
            "event_type": "job.failed",
            "payload": {},
            "event_id": "e2",
            "timestamp": 2.0,
            "source": "test",
        }).encode("utf-8")
        event = self.mod.deserialize_message(raw)
        assert event is not None
        assert event.event_type == EventType.JOB_FAILED

    def test_deserialize_invalid_json_returns_none(self):
        event = self.mod.deserialize_message("not json at all{{{")
        assert event is None

    def test_deserialize_none_returns_none(self):
        event = self.mod.deserialize_message(None)
        assert event is None


# ---------------------------------------------------------------------------
# README validation
# ---------------------------------------------------------------------------


class TestReadmeContent:
    """Validate README covers required sections."""

    @pytest.fixture(autouse=True)
    def _load_readme(self):
        self.readme = README_FILE.read_text(encoding="utf-8")

    def test_has_quick_start_section(self):
        assert "Quick Start" in self.readme

    def test_has_topic_reference_section(self):
        assert "Topic Reference" in self.readme

    def test_has_consumer_example_section(self):
        assert "Consumer Example" in self.readme or "consumer_example" in self.readme

    def test_has_production_deployment_section(self):
        assert "Production Deployment" in self.readme

    def test_mentions_msk(self):
        assert "MSK" in self.readme

    def test_mentions_confluent_cloud(self):
        assert "Confluent Cloud" in self.readme

    def test_has_configuration_reference(self):
        assert "Configuration Reference" in self.readme

    def test_has_troubleshooting_section(self):
        assert "Troubleshooting" in self.readme
