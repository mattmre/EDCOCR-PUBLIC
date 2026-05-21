# Kafka Event Bus for EDCOCR

Production-ready Kafka deployment configuration for the EDCOCR event bus.
Uses KRaft mode (no ZooKeeper dependency) with Bitnami Kafka 3.7.

## Quick Start (Local Development)

```bash
# Start Kafka broker + UI
docker compose -f kafka/docker-compose.kafka.yml up -d

# Verify topics were created
docker exec ocr-kafka kafka-topics.sh \
  --bootstrap-server localhost:9092 --list

# Open Kafka UI
open http://localhost:8080
```

### With Coordinator Stack

```bash
# Start everything together (shared network)
docker compose -f coordinator/docker-compose.coordinator.yml \
               -f kafka/docker-compose.kafka.yml up -d
```

## Topic Reference

| Topic | Partitions | Retention | Key | Description |
|-------|-----------|-----------|-----|-------------|
| `ocr.jobs.submitted` | 3 | 7 days | `job_id` | New job submitted via API or watcher |
| `ocr.jobs.completed` | 3 | 7 days | `job_id` | Job finished with all pages processed |
| `ocr.jobs.failed` | 3 | 30 days | `job_id` | Job failed permanently |
| `ocr.pages.processed` | 6 | 3 days | `job_id` | Per-page OCR completion with confidence |
| `ocr.entities.extracted` | 3 | 30 days | `job_id` | NER/extraction results for a document |
| `ocr.custody.events` | 3 | 90 days | `job_id` | Chain-of-custody forensic audit trail (compacted) |

### Topic Naming Convention

Topics follow the pattern `ocr.<domain>.<action>`:
- `ocr.jobs.*` -- Job lifecycle events
- `ocr.pages.*` -- Page-level processing events
- `ocr.entities.*` -- Entity extraction events
- `ocr.custody.*` -- Forensic audit events

### Message Format

All messages are JSON-serialized `Event` objects from `api/event_bus.py`:

```json
{
  "event_type": "job.completed",
  "payload": {
    "job_id": "abc-123",
    "total_pages": 42,
    "output_path": "/shared/jobs/abc-123/output"
  },
  "event_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": 1710500000.123,
  "source": "gpu-worker-1"
}
```

## Consumer Example

The included `consumer_example.py` demonstrates consuming OCR events with
automatic backend selection (confluent-kafka or kafka-python).

### Install a Kafka client library

```bash
# Option A: confluent-kafka (recommended for production)
pip install confluent-kafka

# Option B: kafka-python (lighter, fewer dependencies)
pip install kafka-python
```

### Run the consumer

```bash
# Default: consume from ocr.jobs.completed
python kafka/consumer_example.py

# Consume from a different topic
python kafka/consumer_example.py --topic ocr.pages.processed

# Batch mode (process 10 messages at a time)
python kafka/consumer_example.py --batch --batch-size 10

# Custom bootstrap server
python kafka/consumer_example.py --bootstrap-servers kafka.prod.internal:9092
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9094` | Kafka broker address(es) |
| `KAFKA_CONSUMER_GROUP` | `ocr-example-consumer` | Consumer group ID |
| `KAFKA_TOPIC` | `ocr.jobs.completed` | Topic to consume from |

## Event Bus Integration

The `api/event_bus.py` module already supports Kafka as a backend. To enable:

```python
from api.event_bus import create_event_bus, Event, EventType

# Create Kafka-backed event bus
bus = create_event_bus(
    "kafka",
    bootstrap_servers="kafka:9092",
    topic_prefix="ocr")

# Publish an event
event = Event(
    event_type=EventType.JOB_COMPLETED,
    payload={"job_id": "abc-123", "total_pages": 10},
    source="api-server")
bus.publish(event)
```

### Configuration via Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EVENT_BUS_BACKEND` | `local` | Backend: `local`, `kafka`, or `sns` |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Kafka broker address(es) |
| `KAFKA_TOPIC_PREFIX` | `ocr` | Prefix for all topic names |

## Production Deployment

### AWS MSK (Managed Streaming for Kafka)

1. Create an MSK cluster with KRaft mode enabled
2. Configure security:
   - Enable TLS encryption in transit
   - Use IAM authentication or SASL/SCRAM
   - Set up VPC security groups
3. Set environment variables:
   ```bash
   KAFKA_BOOTSTRAP_SERVERS=b-1.msk-cluster.abcdef.kafka.us-east-1.amazonaws.com:9096
   KAFKA_SECURITY_PROTOCOL=SASL_SSL
   KAFKA_SASL_MECHANISM=AWS_MSK_IAM
   ```
4. Create topics using MSK CLI or init-topics.sh with appropriate replication factor:
   ```bash
   KAFKA_REPLICATION_FACTOR=3 KAFKA_BOOTSTRAP_SERVER=<msk-broker> bash kafka/init-topics.sh
   ```

### Confluent Cloud

1. Create a Confluent Cloud cluster
2. Generate API keys for producer/consumer
3. Set environment variables:
   ```bash
   KAFKA_BOOTSTRAP_SERVERS=pkc-xxxxx.us-east-1.aws.confluent.cloud:9092
   KAFKA_SECURITY_PROTOCOL=SASL_SSL
   KAFKA_SASL_MECHANISM=PLAIN
   KAFKA_SASL_USERNAME=<api-key>
   KAFKA_SASL_PASSWORD=<api-secret>
   ```

### Self-Hosted Production

For self-hosted Kafka clusters, apply these production settings:

```yaml
# In kafka-topics.yaml, override for production:
replication_factor: 3        # Minimum for HA
min.insync.replicas: 2       # Prevent data loss

# Broker config (server.properties or env vars):
KAFKA_CFG_MIN_INSYNC_REPLICAS: "2"
KAFKA_CFG_DEFAULT_REPLICATION_FACTOR: "3"
KAFKA_CFG_UNCLEAN_LEADER_ELECTION_ENABLE: "false"
```

### Kubernetes (Helm Values)

If deploying alongside the EDCOCR Helm chart, add Kafka as a
StatefulSet or use Strimzi Kafka Operator:

```yaml
# values-kafka.yaml
kafka:
  enabled: true
  replicas: 3
  config:
    offsets.topic.replication.factor: 3
    transaction.state.log.replication.factor: 3
    min.insync.replicas: 2
```

## Configuration Reference

### Docker Compose Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `KAFKA_EXTERNAL_PORT` | `9094` | Host-mapped port for external clients |
| `KAFKA_INTERNAL_PORT` | `9092` | Host-mapped port for internal clients |
| `KAFKA_UI_PORT` | `8080` | Kafka UI web interface port |

### Init Script Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `KAFKA_BOOTSTRAP_SERVER` | `kafka:9092` | Bootstrap server for topic creation |
| `KAFKA_TOPIC_PREFIX` | *(empty)* | Optional prefix for all topic names |
| `KAFKA_REPLICATION_FACTOR` | *(from YAML)* | Override replication factor for all topics |
| `KAFKA_TOPICS_FILE` | `/kafka-topics.yaml` | Path to topic definitions |

## Troubleshooting

### Broker not starting

Check logs for KRaft initialization errors:
```bash
docker logs ocr-kafka 2>&1 | head -50
```

### Topics not created

Re-run the init container:
```bash
docker compose -f kafka/docker-compose.kafka.yml run --rm kafka-init
```

### Consumer not receiving messages

1. Verify the topic exists: `docker exec ocr-kafka kafka-topics.sh --bootstrap-server localhost:9092 --describe --topic ocr.jobs.completed`
2. Check consumer group lag: `docker exec ocr-kafka kafka-consumer-groups.sh --bootstrap-server localhost:9092 --describe --group ocr-example-consumer`
3. Produce a test message: `echo '{"event_type":"job.completed","payload":{"job_id":"test"},"event_id":"test-1","timestamp":0,"source":"test"}' | docker exec -i ocr-kafka kafka-console-producer.sh --bootstrap-server localhost:9092 --topic ocr.jobs.completed`
