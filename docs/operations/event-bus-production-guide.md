# Event Bus Production Guide

Production deployment and operations reference for the EDCOCR event bus
(`api/event_bus.py`). Covers Kafka, AWS SNS/SQS, and local (disabled) backends.

---

## 1. Overview

The event bus publishes OCR job lifecycle events so that downstream systems
(search indexers, notification services, data warehouses, audit archivers) can
react to pipeline activity without polling the API.

### Event types

| EventType | Value | Description |
|-----------|-------|-------------|
| `JOB_SUBMITTED` | `job.submitted` | A new OCR job was submitted via the REST API or file watcher |
| `JOB_STARTED` | `job.started` | A worker picked up the job and began processing |
| `JOB_COMPLETED` | `job.completed` | All pages processed successfully |
| `JOB_FAILED` | `job.failed` | Job failed permanently (retries exhausted) |
| `JOB_CANCELLED` | `job.cancelled` | Job was cancelled by user or operator |
| `PAGE_PROCESSED` | `page.processed` | A single page completed OCR (text + confidence) |
| `WORKER_ONLINE` | `worker.online` | A pipeline worker registered or sent a heartbeat |
| `WORKER_OFFLINE` | `worker.offline` | A worker missed its heartbeat and was marked offline |
| `ALERT_TRIGGERED` | `alert.triggered` | A monitoring alert fired (queue depth, error rate, etc.) |
| `CUSTOM` | `custom` | Application-defined event (extension point) |

### Backend selection

Set `EVENT_BUS_BACKEND` to choose the messaging backend:

| Value | Backend | When to use |
|-------|---------|-------------|
| `local` (default) | In-process callbacks | Development, single-node deployments, or when no external consumers exist |
| `kafka` | Apache Kafka | Distributed deployments needing durable, replayable event streams |
| `sns` | AWS SNS/SQS | AWS-native deployments using serverless consumers (Lambda, ECS tasks) |

The factory function `create_event_bus` reads `EVENT_BUS_BACKEND` from the
environment and returns the appropriate implementation. All backends implement
the same `EventBus` abstract interface (`publish`, `subscribe`, `unsubscribe`).

---

## 2. Kafka Production Configuration

### 2.1 Required environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EVENT_BUS_BACKEND` | `local` | Set to `kafka` to enable Kafka publishing |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Comma-separated broker addresses |
| `KAFKA_TOPIC_PREFIX` | `ocr-pipeline` | Prefix for topic names (only used when `use_topic_map=False`) |
| `KAFKA_SECURITY_PROTOCOL` | *(empty, PLAINTEXT)* | `PLAINTEXT`, `SSL`, `SASL_PLAINTEXT`, or `SASL_SSL` |
| `KAFKA_SASL_MECHANISM` | *(empty)* | `PLAIN`, `SCRAM-SHA-256`, `SCRAM-SHA-512`, or `AWS_MSK_IAM` |
| `KAFKA_SASL_USERNAME` | *(empty)* | SASL username (Confluent Cloud API key, SCRAM user) |
| `KAFKA_SASL_PASSWORD` | *(empty)* | SASL password (Confluent Cloud API secret, SCRAM password) |

Per-topic name overrides use the pattern `KAFKA_TOPIC_<EVENT_TYPE>`. For
example, `KAFKA_TOPIC_JOB_COMPLETED=myapp.jobs.done` overrides the topic used
for `job.completed` events.

### 2.2 Topic naming conventions

Topics follow the pattern `ocr.<domain>.<action>`:

| Topic | Maps to event types | Partitions (default) | Retention |
|-------|---------------------|----------------------|-----------|
| `ocr.jobs.submitted` | `job.submitted`, `job.started`, `worker.online`, `worker.offline`, `alert.triggered`, `custom` | 3 | 7 days |
| `ocr.jobs.completed` | `job.completed` | 3 | 7 days |
| `ocr.jobs.failed` | `job.failed`, `job.cancelled` | 3 | 30 days |
| `ocr.pages.processed` | `page.processed` | 6 | 3 days |
| `ocr.entities.extracted` | *(emitted by NER/extraction pipeline)* | 3 | 30 days |
| `ocr.custody.events` | *(emitted by chain-of-custody module)* | 3 | 90 days (compacted) |

The `DEFAULT_TOPIC_MAP` in `api/event_bus.py` defines the mapping from event
type values to canonical topic names. Multiple event types can share a topic
(for example, `job.submitted` and `job.started` both publish to
`ocr.jobs.submitted`).

Topic definitions live in `kafka/kafka-topics.yaml` and are created
automatically by `kafka/init-topics.sh` when using the Docker Compose overlay.

### 2.3 Partition count recommendations

Partition count determines maximum consumer parallelism for a topic. Each
partition is consumed by at most one consumer in a group.

| Scenario | `ocr.jobs.*` | `ocr.pages.processed` |
|----------|-------------|----------------------|
| Development / single node | 1-3 | 3 |
| Small cluster (1-4 GPU workers) | 3 | 6 |
| Medium cluster (5-12 GPU workers) | 6 | 12 |
| Large cluster (13+ GPU workers) | 12 | 24 |

The `ocr.pages.processed` topic has higher throughput (one event per page) and
should always have more partitions than the job-level topics.

Messages are keyed by `job_id`, which ensures all events for a single job land
on the same partition and are consumed in order.

### 2.4 Replication factor

| Environment | Replication factor | `min.insync.replicas` |
|-------------|-------------------|-----------------------|
| Development | 1 | 1 |
| Staging | 2 | 1 |
| Production | 3 | 2 |

Production clusters must set `unclean.leader.election.enable=false` to prevent
data loss during broker failures.

Override the replication factor for all topics at creation time:

```bash
KAFKA_REPLICATION_FACTOR=3 bash kafka/init-topics.sh
```

### 2.5 Retention policy

Events are append-only audit records. Recommended retention periods:

| Topic | Retention | Rationale |
|-------|-----------|-----------|
| `ocr.jobs.submitted` | 7 days | Short-lived coordination events |
| `ocr.jobs.completed` | 7 days | Downstream consumers should process within days |
| `ocr.jobs.failed` | 30 days | Failure investigation window |
| `ocr.pages.processed` | 3 days | High-volume, short-lived progress events |
| `ocr.entities.extracted` | 30 days | Entity data needed for compliance review |
| `ocr.custody.events` | 90 days | Forensic audit trail; uses `compact,delete` cleanup |

For long-term archival, configure a Kafka Connect sink (S3 Sink Connector,
Elasticsearch Sink Connector) to persist events beyond retention.

### 2.6 Consumer group configuration

Each downstream service should use a distinct `group.id`:

```
ocr-search-indexer          # Updates Elasticsearch/OpenSearch
ocr-notification-service    # Sends email/Slack alerts
ocr-billing-tracker         # Increments per-tenant usage counters
ocr-audit-archiver          # Writes events to long-term S3 storage
```

A single consumer group can have up to N consumers, where N is the partition
count for the subscribed topic. Additional consumers beyond N sit idle as
standby.

Recommended consumer settings:

| Setting | Value | Rationale |
|---------|-------|-----------|
| `auto.offset.reset` | `earliest` | Process all events after first deployment |
| `enable.auto.commit` | `true` | Simplifies offset management for at-least-once delivery |
| `auto.commit.interval.ms` | `5000` | 5-second commit interval balances latency and throughput |
| `session.timeout.ms` | `30000` | Detect dead consumers within 30 seconds |
| `max.poll.interval.ms` | `300000` | Allow up to 5 minutes between polls for slow processing |

### 2.7 SASL/SSL authentication

#### Confluent Cloud

```bash
EVENT_BUS_BACKEND=kafka
KAFKA_BOOTSTRAP_SERVERS=pkc-xxxxx.us-east-1.aws.confluent.cloud:9092
KAFKA_SECURITY_PROTOCOL=SASL_SSL
KAFKA_SASL_MECHANISM=PLAIN
KAFKA_SASL_USERNAME=<confluent-api-key>
KAFKA_SASL_PASSWORD=<confluent-api-secret>
```

#### AWS MSK with IAM authentication

```bash
EVENT_BUS_BACKEND=kafka
KAFKA_BOOTSTRAP_SERVERS=b-1.msk-cluster.region.amazonaws.com:9098
KAFKA_SECURITY_PROTOCOL=SASL_SSL
KAFKA_SASL_MECHANISM=AWS_MSK_IAM
```

IAM authentication uses the AWS SDK credential chain (environment variables,
instance profile, or ECS task role). No username/password is needed.

#### Self-hosted with SCRAM-SHA-256

```bash
EVENT_BUS_BACKEND=kafka
KAFKA_BOOTSTRAP_SERVERS=kafka-1.internal:9093,kafka-2.internal:9093
KAFKA_SECURITY_PROTOCOL=SASL_SSL
KAFKA_SASL_MECHANISM=SCRAM-SHA-256
KAFKA_SASL_USERNAME=ocr-producer
KAFKA_SASL_PASSWORD=<scram-password>
```

### 2.8 Local Kafka (Docker Compose, KRaft mode)

The project includes a ready-to-use Docker Compose overlay at
`kafka/docker-compose.kafka.yml`. It runs a single Kafka broker in KRaft mode
(no ZooKeeper) with a topic initialization sidecar and Kafka UI.

```bash
# Standalone Kafka
docker compose -f kafka/docker-compose.kafka.yml up -d

# With the coordinator stack (shared ocr-network)
docker compose -f coordinator/docker-compose.coordinator.yml \
               -f kafka/docker-compose.kafka.yml up -d

# Verify topics
docker exec ocr-kafka kafka-topics.sh \
  --bootstrap-server localhost:9092 --list

# Open Kafka UI at http://localhost:8080
```

Services started:

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `kafka` | `bitnami/kafka:3.7` | 9092 (internal), 9094 (external) | KRaft broker |
| `kafka-init` | `bitnami/kafka:3.7` | -- | One-shot topic creator |
| `kafka-ui` | `provectuslabs/kafka-ui:v0.7.2` | 8080 | Web-based topic/message browser |

Port mappings are configurable via environment variables:

```bash
KAFKA_EXTERNAL_PORT=9094    # Host port for external Kafka clients
KAFKA_INTERNAL_PORT=9092    # Host port for internal Kafka clients
KAFKA_UI_PORT=8080          # Kafka UI web interface
```

---

## 3. AWS SNS/SQS Configuration

### 3.1 Required environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EVENT_BUS_BACKEND` | `local` | Set to `sns` to enable SNS publishing |
| `SNS_REGION` | `us-east-1` | AWS region for SNS topics |
| `SNS_TOPIC_ARN_PREFIX` | *(empty)* | ARN prefix, e.g. `arn:aws:sns:us-east-1:123456789012` |

The `SnsEventBus` constructs topic ARNs by appending the event type value to
the prefix: `{SNS_TOPIC_ARN_PREFIX}:{event_type.value}`.

For example, a `job.completed` event publishes to:
```
arn:aws:sns:us-east-1:123456789012:job.completed
```

### 3.2 IAM policy requirements

The API server (or ECS task, Lambda function) needs an IAM policy that grants
`sns:Publish` to the relevant topic ARNs.

Producer policy (attach to the API server role):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "OcrEventBusPublish",
      "Effect": "Allow",
      "Action": "sns:Publish",
      "Resource": "arn:aws:sns:us-east-1:123456789012:job.*"
    },
    {
      "Sid": "OcrEventBusPublishPages",
      "Effect": "Allow",
      "Action": "sns:Publish",
      "Resource": "arn:aws:sns:us-east-1:123456789012:page.*"
    }
  ]
}
```

Consumer policy (attach to downstream consumer roles):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "OcrEventBusConsume",
      "Effect": "Allow",
      "Action": [
        "sqs:ReceiveMessage",
        "sqs:DeleteMessage",
        "sqs:GetQueueAttributes",
        "sqs:ChangeMessageVisibility"
      ],
      "Resource": "arn:aws:sqs:us-east-1:123456789012:ocr-*"
    }
  ]
}
```

### 3.3 SNS topic + SQS queue + subscription setup

CloudFormation snippet for a single event type (`job.completed`). Repeat for
each event type you need to consume.

```yaml
AWSTemplateFormatVersion: "2010-09-09"
Description: EDCOCR event bus SNS/SQS infrastructure

Parameters:
  Environment:
    Type: String
    Default: production
    AllowedValues: [development, staging, production]

Resources:
  # SNS Topic
  JobCompletedTopic:
    Type: AWS::SNS::Topic
    Properties:
      TopicName: !Sub "ocr-${Environment}-job-completed"
      Tags:
        - Key: Project
          Value: ocr-local
        - Key: Environment
          Value: !Ref Environment

  # SQS Queue (consumer)
  JobCompletedQueue:
    Type: AWS::SQS::Queue
    Properties:
      QueueName: !Sub "ocr-${Environment}-job-completed"
      VisibilityTimeout: 300           # 5 minutes for processing
      MessageRetentionPeriod: 1209600  # 14 days
      ReceiveMessageWaitTimeSeconds: 20  # Long polling
      RedrivePolicy:
        deadLetterTargetArn: !GetAtt JobCompletedDLQ.Arn
        maxReceiveCount: 3
      Tags:
        - Key: Project
          Value: ocr-local

  # Dead-letter queue
  JobCompletedDLQ:
    Type: AWS::SQS::Queue
    Properties:
      QueueName: !Sub "ocr-${Environment}-job-completed-dlq"
      MessageRetentionPeriod: 1209600  # 14 days
      Tags:
        - Key: Project
          Value: ocr-local

  # SQS policy allowing SNS to deliver messages
  JobCompletedQueuePolicy:
    Type: AWS::SQS::QueuePolicy
    Properties:
      Queues:
        - !Ref JobCompletedQueue
      PolicyDocument:
        Statement:
          - Sid: AllowSNSDelivery
            Effect: Allow
            Principal:
              Service: sns.amazonaws.com
            Action: sqs:SendMessage
            Resource: !GetAtt JobCompletedQueue.Arn
            Condition:
              ArnEquals:
                aws:SourceArn: !Ref JobCompletedTopic

  # SNS -> SQS subscription
  JobCompletedSubscription:
    Type: AWS::SNS::Subscription
    Properties:
      TopicArn: !Ref JobCompletedTopic
      Protocol: sqs
      Endpoint: !GetAtt JobCompletedQueue.Arn
      RawMessageDelivery: true  # Deliver raw JSON, not wrapped in SNS envelope

Outputs:
  TopicArn:
    Value: !Ref JobCompletedTopic
  QueueUrl:
    Value: !Ref JobCompletedQueue
  DLQUrl:
    Value: !Ref JobCompletedDLQ
```

### 3.4 Dead-letter queue setup

The CloudFormation snippet above includes a DLQ with `maxReceiveCount: 3`.
Messages that fail processing three times are moved to the DLQ automatically.

DLQ monitoring recommendations:

- Set a CloudWatch alarm on `ApproximateNumberOfMessagesVisible` for the DLQ
- Threshold: >= 1 message for 5 minutes triggers a notification
- Action: SNS notification to an operations email/PagerDuty endpoint

### 3.5 Message retention and visibility timeout

| Setting | Recommended value | Rationale |
|---------|-------------------|-----------|
| `VisibilityTimeout` | 300 seconds (5 min) | Allow time for downstream processing |
| `MessageRetentionPeriod` | 1,209,600 seconds (14 days) | Buffer for consumer outages |
| `ReceiveMessageWaitTimeSeconds` | 20 seconds | Long polling reduces API calls and cost |
| `maxReceiveCount` (redrive) | 3 | Move poison messages to DLQ after 3 failures |

---

## 4. Event Schema Reference

### 4.1 Event envelope

Every event, regardless of backend, uses the same JSON envelope:

```json
{
  "event_type": "job.completed",
  "payload": { ... },
  "event_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": 1710500000.123,
  "source": "gpu-worker-1"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `event_type` | `string` | One of the `EventType` values (e.g. `job.submitted`) |
| `payload` | `object` | Event-specific data (see below) |
| `event_id` | `string` | UUID v4, unique per event instance |
| `timestamp` | `float` | Unix epoch seconds (with fractional milliseconds) |
| `source` | `string` | Identifier of the component that emitted the event |

### 4.2 Payload schemas by event type

#### `job.submitted`

```json
{
  "job_id": "job_a1b2c3d4e5f6",
  "filename": "contract-2024.pdf",
  "source": "api",
  "total_pages": 42,
  "options": {
    "language": "en",
    "enable_docintel": false,
    "dpi": 300
  }
}
```

#### `job.started`

```json
{
  "job_id": "job_a1b2c3d4e5f6",
  "worker_id": "gpu-worker-1",
  "started_at": 1710500100.0
}
```

#### `job.completed`

```json
{
  "job_id": "job_a1b2c3d4e5f6",
  "total_pages": 42,
  "output_path": "/shared/jobs/job_a1b2c3d4e5f6/output",
  "processing_time_seconds": 127.5,
  "avg_confidence": 0.94
}
```

#### `job.failed`

```json
{
  "job_id": "job_a1b2c3d4e5f6",
  "error": "GPU OOM during page 17",
  "failed_page": 17,
  "retries_exhausted": true
}
```

#### `job.cancelled`

```json
{
  "job_id": "job_a1b2c3d4e5f6",
  "cancelled_by": "operator",
  "reason": "Duplicate submission"
}
```

#### `page.processed`

```json
{
  "job_id": "job_a1b2c3d4e5f6",
  "page_number": 17,
  "confidence": 0.92,
  "language_detected": "en",
  "word_count": 347,
  "processing_time_ms": 1250
}
```

#### `worker.online`

```json
{
  "worker_id": "gpu-worker-1",
  "capabilities": ["ocr_gpu", "docintel"],
  "gpu_model": "NVIDIA RTX 4090",
  "vram_mb": 24576
}
```

#### `worker.offline`

```json
{
  "worker_id": "gpu-worker-1",
  "last_heartbeat": 1710500050.0,
  "reason": "heartbeat_timeout"
}
```

#### `alert.triggered`

```json
{
  "alert_name": "queue_depth_critical",
  "severity": "critical",
  "metric_value": 150,
  "threshold": 100,
  "message": "OCR job queue exceeds 100 pending jobs"
}
```

### 4.3 Event ordering guarantees

**Kafka**: Events for the same `job_id` are ordered within a partition because
`job_id` is used as the message key. Events across different jobs may arrive
out of order relative to each other. There is no global ordering guarantee.

**SNS/SQS**: Standard SQS queues provide best-effort ordering. If strict
per-job ordering is required, use SQS FIFO queues with `job_id` as the
`MessageGroupId`.

**Local**: Events are delivered synchronously in publish order within the same
thread. Callbacks execute sequentially.

### 4.4 Delivery semantics

**Kafka**: At-least-once delivery. Consumers should be idempotent (see
Section 5.3).

**SNS/SQS**: At-least-once delivery. SQS standard queues may deliver messages
more than once.

**Local**: Exactly-once within the process. No persistence across restarts.

---

## 5. Downstream Consumer Patterns

### 5.1 Kafka consumer (confluent-kafka)

The project includes a complete consumer example at `kafka/consumer_example.py`
with automatic backend selection (confluent-kafka preferred, kafka-python
fallback), signal handling, and both single-message and batch modes.

Minimal standalone consumer:

```python
"""Minimal Kafka consumer for EDCOCR events."""

import json
import signal
import sys

from confluent_kafka import Consumer

running = True


def shutdown(signum, frame):
    global running
    running = False


signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)


def main:
    consumer = Consumer({
        "bootstrap.servers": "kafka:9092",
        "group.id": "my-downstream-service",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": True,
    })
    consumer.subscribe(["ocr.jobs.completed"])

    processed_ids = set  # Deduplication window (see Section 5.3)

    try:
        while running:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error:
                print(f"Consumer error: {msg.error}", file=sys.stderr)
                continue

            event = json.loads(msg.value.decode("utf-8"))
            event_id = event.get("event_id", "")

            # Idempotency check
            if event_id in processed_ids:
                continue
            processed_ids.add(event_id)

            # --- Your processing logic here ---
            job_id = event["payload"].get("job_id", "unknown")
            print(f"Job completed: {job_id}")

    finally:
        consumer.close


if __name__ == "__main__":
    main
```

### 5.2 SQS consumer (boto3)

```python
"""Minimal SQS consumer for EDCOCR events via SNS->SQS subscription."""

import json
import signal
import sys

import boto3

running = True


def shutdown(signum, frame):
    global running
    running = False


signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)


def main:
    sqs = boto3.client("sqs", region_name="us-east-1")
    queue_url = "https://sqs.us-east-1.amazonaws.com/123456789012/ocr-production-job-completed"

    processed_ids = set

    while running:
        response = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=10,
            WaitTimeSeconds=20,         # Long polling
            VisibilityTimeout=300)

        messages = response.get("Messages", [])
        for msg in messages:
            try:
                event = json.loads(msg["Body"])
                event_id = event.get("event_id", "")

                # Idempotency check
                if event_id in processed_ids:
                    sqs.delete_message(
                        QueueUrl=queue_url,
                        ReceiptHandle=msg["ReceiptHandle"])
                    continue
                processed_ids.add(event_id)

                # --- Your processing logic here ---
                job_id = event["payload"].get("job_id", "unknown")
                print(f"Job completed: {job_id}")

                # Delete message after successful processing
                sqs.delete_message(
                    QueueUrl=queue_url,
                    ReceiptHandle=msg["ReceiptHandle"])

            except (json.JSONDecodeError, KeyError) as exc:
                print(f"Failed to process message: {exc}", file=sys.stderr)
                # Message will become visible again after VisibilityTimeout
                # and eventually go to DLQ after maxReceiveCount

    print("Consumer stopped.")


if __name__ == "__main__":
    main
```

### 5.3 Idempotency considerations

Events are delivered at-least-once on both Kafka and SNS/SQS. Consumers must
handle duplicate deliveries.

**Deduplication key**: Use `event_id` (UUID v4, unique per event) as the
primary deduplication key. For job-level idempotency, `job_id` +
`event_type` can also serve as a composite key.

**Strategies**:

1. **In-memory set** -- Suitable for single-instance consumers. Keep a
   bounded set of recently seen `event_id` values (e.g. last 10,000).

2. **Database upsert** -- For durable deduplication, store `event_id` in a
   database table with a unique constraint. Use `INSERT ... ON CONFLICT DO
   NOTHING` (PostgreSQL) or equivalent.

3. **Redis TTL key** -- Set a Redis key `event:{event_id}` with a TTL
   matching the topic retention period. Check existence before processing.

4. **Kafka consumer offsets** -- If processing is stateless and fast, relying
   on Kafka's committed offsets (with `enable.auto.commit=True`) is often
   sufficient.

### 5.4 Error handling and retry

- **Transient failures** (network, temporary DB outage): Do not delete/commit
  the message. The message will be redelivered after `VisibilityTimeout` (SQS)
  or rebalance (Kafka).

- **Permanent failures** (malformed event, missing required field): Log the
  error and delete/commit the message to prevent infinite retries. Consider
  publishing to a dead-letter topic or table for investigation.

- **Poison messages**: SQS DLQ handles this automatically after
  `maxReceiveCount`. For Kafka, implement a manual DLQ by publishing failed
  messages to a `*.dlq` topic after N processing attempts.

---

## 6. Monitoring

### 6.1 Key metrics to monitor

| Metric | Source | Alert threshold |
|--------|--------|-----------------|
| Consumer group lag | Kafka broker / consumer group | > 1000 messages for 5 min |
| Publish error rate | Application logs (`Kafka publish failed`) | > 0 for 2 min |
| Consumer rebalance frequency | Consumer logs | > 2 rebalances per hour |
| SQS `ApproximateNumberOfMessagesVisible` | CloudWatch | > 100 for 5 min |
| SQS DLQ message count | CloudWatch | > 0 for 5 min |
| Event processing latency | Custom (timestamp delta) | p99 > 30 seconds |

### 6.2 Prometheus metrics

The event bus module itself does not currently expose Prometheus metrics
directly. To monitor event bus health via Prometheus:

1. **Kafka exporter**: Deploy
   [kafka-exporter](https://github.com/danielqsj/kafka_exporter) alongside
   your Kafka cluster. It exposes consumer group lag, topic partition offsets,
   and broker metrics.

   ```yaml
   # Add to docker-compose.kafka.yml
   kafka-exporter:
     image: danielqsj/kafka-exporter:latest
     command:
       - "--kafka.server=kafka:9092"
     ports:
       - "9308:9308"
   ```

2. **Application-level counters**: The coordinator's Prometheus endpoint
   (`/api/v1/prometheus/`) already tracks job lifecycle metrics. Correlate
   job counts with event bus publish counts.

3. **SQS CloudWatch**: Use the CloudWatch metrics exporter for Prometheus or
   the AWS-native CloudWatch alarms described in Section 3.4.

### 6.3 Alerting recommendations

| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| Consumer lag critical | Lag > 5000 for 10 min | Critical | Scale consumer instances |
| Publish failures | Any `Kafka publish failed` log | Warning | Check broker connectivity |
| DLQ non-empty | SQS DLQ messages > 0 | Warning | Investigate failed messages |
| Consumer group empty | 0 active consumers for 5 min | Critical | Restart consumer service |
| Topic under-replicated | ISR < replication factor | Critical | Check broker health |

### 6.4 Kafka UI

The Docker Compose overlay includes Kafka UI at `http://localhost:8080`. Use
it to:

- Browse topics and inspect messages
- Monitor consumer group lag in real time
- View broker configuration
- Produce test messages for debugging

---

## 7. Disabling the Event Bus

### 7.1 How to run without an event bus

The event bus is disabled by default. When `EVENT_BUS_BACKEND` is unset or set
to `local`, the `LocalEventBus` is used. This is an in-process implementation
that dispatches events to registered callbacks within the same Python process.

To explicitly disable:

```bash
EVENT_BUS_BACKEND=local  # or simply do not set the variable
```

No Kafka, SNS, or any external messaging dependency is required.

### 7.2 Impact on functionality

When using the local backend:

- **No external consumers**: Downstream services cannot subscribe to events
  unless they are running in the same process.
- **No persistence**: Events are lost on process restart. The `LocalEventBus`
  keeps an in-memory history (last 1,000 events) for debugging, but this is
  not persisted to disk.
- **No replay**: The event replay endpoint (`GET /api/v1/jobs/{job_id}/events`)
  uses the separate `EventStore` (SQLite-backed), which is independent of the
  event bus backend and continues to work regardless of backend choice.
- **No impact on core OCR**: The pipeline processes documents identically
  regardless of event bus backend. The event bus is an observation layer, not
  a control-plane dependency.
- **Webhooks still work**: Webhook delivery (`api/webhooks.py`) is independent
  of the event bus and functions on all backends.

### 7.3 Migrating from local to Kafka/SNS

1. Deploy the Kafka or SNS infrastructure (see Sections 2 and 3).
2. Set `EVENT_BUS_BACKEND=kafka` (or `sns`) in your environment.
3. Restart the API server. Events will begin publishing immediately.
4. Deploy consumer services to process the backlog.

No data migration is needed. The event bus is fire-and-forget from the
producer's perspective -- historical events before the backend switch are
available only through the `EventStore` replay endpoint, not through Kafka
or SQS.

---

## Appendix A: Environment Variable Quick Reference

| Variable | Backend | Default | Description |
|----------|---------|---------|-------------|
| `EVENT_BUS_BACKEND` | All | `local` | Backend: `local`, `kafka`, `sns` |
| `KAFKA_BOOTSTRAP_SERVERS` | Kafka | `localhost:9092` | Broker addresses |
| `KAFKA_TOPIC_PREFIX` | Kafka | `ocr-pipeline` | Topic name prefix (legacy mode) |
| `KAFKA_SECURITY_PROTOCOL` | Kafka | *(empty)* | `PLAINTEXT`, `SSL`, `SASL_PLAINTEXT`, `SASL_SSL` |
| `KAFKA_SASL_MECHANISM` | Kafka | *(empty)* | `PLAIN`, `SCRAM-SHA-256`, `SCRAM-SHA-512`, `AWS_MSK_IAM` |
| `KAFKA_SASL_USERNAME` | Kafka | *(empty)* | SASL username |
| `KAFKA_SASL_PASSWORD` | Kafka | *(empty)* | SASL password |
| `KAFKA_TOPIC_<EVENT>` | Kafka | *(from map)* | Per-event topic override (e.g. `KAFKA_TOPIC_JOB_COMPLETED`) |
| `SNS_REGION` | SNS | `us-east-1` | AWS region |
| `SNS_TOPIC_ARN_PREFIX` | SNS | *(empty)* | ARN prefix for SNS topics |
| `KAFKA_EXTERNAL_PORT` | Docker | `9094` | Host port for external Kafka clients |
| `KAFKA_INTERNAL_PORT` | Docker | `9092` | Host port for internal Kafka clients |
| `KAFKA_UI_PORT` | Docker | `8080` | Kafka UI web interface port |

## Appendix B: File Reference

| File | Purpose |
|------|---------|
| `api/event_bus.py` | Event bus abstraction (LocalEventBus, KafkaEventBus, SnsEventBus) |
| `kafka/docker-compose.kafka.yml` | Docker Compose overlay for local Kafka (KRaft) |
| `kafka/kafka-topics.yaml` | Topic definitions (partitions, retention, config) |
| `kafka/init-topics.sh` | Topic creation script (runs as init container) |
| `kafka/consumer_example.py` | Reference consumer with confluent-kafka/kafka-python fallback |
| `kafka/README.md` | Kafka-specific quick start and deployment reference |
| `api/routers/events.py` | Event replay and webhook DLQ API endpoints |
| `tests/test_event_bus.py` | Unit tests for event bus abstraction |
| `tests/test_kafka_config.py` | Tests for Kafka deployment configuration |
