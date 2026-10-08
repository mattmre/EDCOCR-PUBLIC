import hashlib
import json
import uuid
from datetime import timedelta

from django.db import migrations, models
from django.utils import timezone


def seed_existing_processing_owners(apps, schema_editor):
    Job = apps.get_model("jobs", "Job")
    now = timezone.now()
    for job in Job.objects.filter(status="processing").exclude(celery_task_id="").iterator():
        # A mixed-version native worker may still be running. Block a new
        # attempt until its bounded legacy lease expires rather than granting
        # a duplicate owner immediately after the schema upgrade.
        identity = hashlib.sha256(json.dumps([
            str(job.job_id), int(job.dispatch_generation), str(job.tenant_id or ""),
            str(job.source_hash or ""), str(job.source_object_key or ""),
            str(job.source_file or ""), str(job.celery_task_id or ""),
        ], separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
        Job.objects.filter(pk=job.pk, processing_attempt_token="").update(
            processing_attempt_token=uuid.uuid4().hex,
            processing_attempt_identity=identity,
            processing_lease_expires_at=(job.started_at or now) + timedelta(minutes=30),
        )


class Migration(migrations.Migration):
    dependencies = [("jobs", "0030_staged_page_ocr")]

    operations = [
        migrations.AddField(
            model_name="job", name="processing_attempt_token",
            field=models.CharField(blank=True, default="", max_length=64),
        ),
        migrations.AddField(
            model_name="job", name="processing_lease_expires_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="job", name="processing_attempt_identity",
            field=models.CharField(blank=True, default="", max_length=64),
        ),
        migrations.RunPython(seed_existing_processing_owners, migrations.RunPython.noop),
    ]
