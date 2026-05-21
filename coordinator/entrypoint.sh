#!/bin/bash
set -e

# Wait for PostgreSQL
echo "Waiting for PostgreSQL..."
while ! pg_isready -h postgres -U ocr -d ocr_coordinator -q 2>/dev/null; do
    sleep 1
done
echo "PostgreSQL ready."

# Run migrations
echo "Running migrations..."
python manage.py migrate --noinput

# Collect static files
python manage.py collectstatic --noinput 2>/dev/null || true

# Execute the main command
exec "$@"
