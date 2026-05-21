#!/usr/bin/env bash
# init-topics.sh — Create Kafka topics from kafka-topics.yaml on startup
#
# This script runs as a one-shot init container after the Kafka broker
# is healthy.  It parses kafka-topics.yaml and creates topics with the
# configured partitions, replication factor, and per-topic config overrides.
#
# Dependencies: bash, grep, sed (available in bitnami/kafka image)
#
# Environment variables:
#   KAFKA_BOOTSTRAP_SERVER — broker address (default: kafka:9092)
#   KAFKA_TOPIC_PREFIX     — optional prefix prepended to topic names
#   KAFKA_REPLICATION_FACTOR — override replication factor for all topics

set -euo pipefail

BOOTSTRAP="${KAFKA_BOOTSTRAP_SERVER:-kafka:9092}"
PREFIX="${KAFKA_TOPIC_PREFIX:-}"
REPLICATION_OVERRIDE="${KAFKA_REPLICATION_FACTOR:-}"
CONFIG_FILE="${KAFKA_TOPICS_FILE:-/kafka-topics.yaml}"

echo "=== EDCOCR Kafka topic initialization ==="
echo "Bootstrap server: ${BOOTSTRAP}"
echo "Topic prefix: ${PREFIX:-<none>}"
echo "Config file: ${CONFIG_FILE}"
echo ""

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Topic config file not found: ${CONFIG_FILE}"
    exit 1
fi

# Wait for broker to be fully ready (beyond healthcheck)
MAX_RETRIES=30
RETRY=0
until kafka-metadata.sh --snapshot /bitnami/kafka/data/__cluster_metadata-0/00000000000000000000.log --cluster-id 2>/dev/null || \
      kafka-topics.sh --bootstrap-server "${BOOTSTRAP}" --list >/dev/null 2>&1; do
    RETRY=$((RETRY + 1))
    if [ "${RETRY}" -ge "${MAX_RETRIES}" ]; then
        echo "ERROR: Kafka broker not ready after ${MAX_RETRIES} retries"
        exit 1
    fi
    echo "Waiting for Kafka broker... (${RETRY}/${MAX_RETRIES})"
    sleep 2
done

echo "Kafka broker is ready."
echo ""

# Simple YAML parser — extract topic definitions
# Format expected:
#   - name: ocr.jobs.submitted
#     partitions: 3
#     replication_factor: 1
#     config:
#       cleanup.policy: delete
#       retention.ms: "604800000"

CURRENT_TOPIC=""
CURRENT_PARTITIONS=""
CURRENT_REPLICATION=""
CURRENT_CONFIGS=""
IN_CONFIG_BLOCK=false
TOPICS_CREATED=0
TOPICS_EXISTED=0

create_topic() {
    local topic_name="$1"
    local partitions="$2"
    local replication="$3"
    local configs="$4"

    # Apply prefix if set
    if [ -n "${PREFIX}" ]; then
        topic_name="${PREFIX}.${topic_name}"
    fi

    # Apply replication override if set
    if [ -n "${REPLICATION_OVERRIDE}" ]; then
        replication="${REPLICATION_OVERRIDE}"
    fi

    # Build config flags
    CONFIG_FLAGS=""
    if [ -n "${configs}" ]; then
        # configs is comma-separated key=value pairs
        IFS=',' read -ra CONFIG_ARRAY <<< "${configs}"
        for cfg in "${CONFIG_ARRAY[@]}"; do
            cfg=$(echo "${cfg}" | sed 's/^ *//;s/ *$//')
            if [ -n "${cfg}" ]; then
                CONFIG_FLAGS="${CONFIG_FLAGS} --config ${cfg}"
            fi
        done
    fi

    # Check if topic already exists
    if kafka-topics.sh --bootstrap-server "${BOOTSTRAP}" --describe --topic "${topic_name}" >/dev/null 2>&1; then
        echo "  [EXISTS] ${topic_name} (partitions=${partitions}, replication=${replication})"
        TOPICS_EXISTED=$((TOPICS_EXISTED + 1))
        return
    fi

    # Create topic
    echo "  [CREATE] ${topic_name} (partitions=${partitions}, replication=${replication})"
    # shellcheck disable=SC2086
    kafka-topics.sh --bootstrap-server "${BOOTSTRAP}" \
        --create \
        --topic "${topic_name}" \
        --partitions "${partitions}" \
        --replication-factor "${replication}" \
        ${CONFIG_FLAGS} \
        2>&1 || {
            echo "  [ERROR] Failed to create topic: ${topic_name}"
            return 1
        }
    TOPICS_CREATED=$((TOPICS_CREATED + 1))
}

# Parse the YAML file line by line
while IFS= read -r line || [ -n "${line}" ]; do
    # Strip carriage returns
    line="${line//$'\r'/}"

    # Skip comments and empty lines
    [[ "${line}" =~ ^[[:space:]]*# ]] && continue
    [[ "${line}" =~ ^[[:space:]]*$ ]] && continue

    # Detect new topic entry
    if echo "${line}" | grep -qE '^\s*- name:'; then
        # Flush previous topic if any
        if [ -n "${CURRENT_TOPIC}" ]; then
            create_topic "${CURRENT_TOPIC}" "${CURRENT_PARTITIONS:-3}" "${CURRENT_REPLICATION:-1}" "${CURRENT_CONFIGS}"
        fi
        CURRENT_TOPIC=$(echo "${line}" | sed 's/.*- name:[[:space:]]*//' | sed 's/[[:space:]]*$//')
        CURRENT_PARTITIONS=""
        CURRENT_REPLICATION=""
        CURRENT_CONFIGS=""
        IN_CONFIG_BLOCK=false
        continue
    fi

    # Parse partitions
    if echo "${line}" | grep -qE '^\s+partitions:'; then
        CURRENT_PARTITIONS=$(echo "${line}" | sed 's/.*partitions:[[:space:]]*//' | sed 's/[[:space:]]*$//')
        IN_CONFIG_BLOCK=false
        continue
    fi

    # Parse replication_factor
    if echo "${line}" | grep -qE '^\s+replication_factor:'; then
        CURRENT_REPLICATION=$(echo "${line}" | sed 's/.*replication_factor:[[:space:]]*//' | sed 's/[[:space:]]*$//')
        IN_CONFIG_BLOCK=false
        continue
    fi

    # Detect config block start
    if echo "${line}" | grep -qE '^\s+config:'; then
        IN_CONFIG_BLOCK=true
        continue
    fi

    # Parse config key-value pairs
    if [ "${IN_CONFIG_BLOCK}" = true ]; then
        # Config lines are indented under config:
        if echo "${line}" | grep -qE '^\s{6,}'; then
            key=$(echo "${line}" | sed 's/^[[:space:]]*//' | cut -d: -f1)
            value=$(echo "${line}" | sed 's/^[^:]*:[[:space:]]*//' | sed 's/^"//;s/"$//' | sed 's/[[:space:]]*$//')
            if [ -n "${key}" ] && [ -n "${value}" ]; then
                if [ -n "${CURRENT_CONFIGS}" ]; then
                    CURRENT_CONFIGS="${CURRENT_CONFIGS},${key}=${value}"
                else
                    CURRENT_CONFIGS="${key}=${value}"
                fi
            fi
        else
            IN_CONFIG_BLOCK=false
        fi
    fi

    # Skip description, key, retention_ms (informational only)
done < "${CONFIG_FILE}"

# Flush last topic
if [ -n "${CURRENT_TOPIC}" ]; then
    create_topic "${CURRENT_TOPIC}" "${CURRENT_PARTITIONS:-3}" "${CURRENT_REPLICATION:-1}" "${CURRENT_CONFIGS}"
fi

echo ""
echo "=== Topic initialization complete ==="
echo "Created: ${TOPICS_CREATED}"
echo "Already existed: ${TOPICS_EXISTED}"
echo "Total: $((TOPICS_CREATED + TOPICS_EXISTED))"
