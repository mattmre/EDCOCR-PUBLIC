#!/bin/sh
# Docker healthcheck script for OCR pipeline.
# Checks that the monitor thread has written a heartbeat within the last 60 seconds.
HEALTHCHECK_FILE="${HEALTHCHECK_FILE:-/app/ocr_healthcheck}"
MAX_AGE=60

if [ ! -f "$HEALTHCHECK_FILE" ]; then
    # Docker's start_period handles startup grace.  Once health evaluation is
    # authoritative, a missing heartbeat means the monitor never started or
    # stopped updating and must fail closed.
    echo "Healthcheck FAILED: heartbeat file is missing"
    exit 1
fi

LAST_BEAT=$(cat "$HEALTHCHECK_FILE" 2>/dev/null | head -1)

# Validate that LAST_BEAT is a numeric timestamp
if ! echo "$LAST_BEAT" | grep -qE '^[0-9]+$'; then
    echo "Healthcheck FAILED: invalid heartbeat value"
    exit 1
fi

NOW=$(date +%s)
AGE=$((NOW - LAST_BEAT))

if [ "$AGE" -gt "$MAX_AGE" ]; then
    echo "Healthcheck FAILED: heartbeat is ${AGE}s old (max ${MAX_AGE}s)"
    exit 1
fi

exit 0
