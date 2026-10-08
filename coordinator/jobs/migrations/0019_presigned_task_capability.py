"""Keep presigned bearer URLs out of Celery task arguments.

Capabilities are short-lived, job-bound rows. Workers receive only the UUID
and resolve the encrypted payload after loading the matching job.
"""

import uuid

import django.db.models.deletion
from django.db import migrations, models

import jobs.encrypted_field


class Migration(migrations.Migration):
    dependencies = [
        ("jobs", "0018_job_fanout_dispatched"),
    ]

    operations = [
        migrations.CreateModel(
            name="PresignedTaskCapability",
            fields=[
                (
                    "capability_id",
                    models.UUIDField(
                        default=uuid.uuid4,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                ("operation", models.CharField(max_length=64)),
                (
                    "payload",
                    jobs.encrypted_field.EncryptedTextField(
                        help_text=(
                            "Encrypted presigned URL map; never serialize this "
                            "field into Celery."
                        ),
                    ),
                ),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("expires_at", models.DateTimeField(db_index=True)),
                (
                    "job",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="presigned_capabilities",
                        to="jobs.job",
                    ),
                ),
            ],
            options={
                "indexes": [
                    models.Index(
                        fields=["job", "operation", "expires_at"],
                        name="jobs_presigned_job_op_exp_idx",
                    ),
                ],
            },
        ),
    ]
