"""Durable coordinator database-to-Celery dispatch contracts."""

import uuid

import django.db.models.deletion
from django.db import migrations, models
from django.utils import timezone


class Migration(migrations.Migration):
    dependencies = [
        ("jobs", "0019_presigned_task_capability"),
    ]

    operations = [
        migrations.AddField(
            model_name="job",
            name="dispatch_generation",
            field=models.PositiveIntegerField(default=0),
        ),
        migrations.CreateModel(
            name="TaskDispatchOutbox",
            fields=[
                (
                    "dispatch_id",
                    models.UUIDField(
                        default=uuid.uuid4,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                ("stage", models.CharField(max_length=64)),
                ("generation", models.CharField(max_length=64)),
                ("task_name", models.CharField(max_length=255)),
                ("args", models.JSONField(blank=True, default=list)),
                ("kwargs", models.JSONField(blank=True, default=dict)),
                ("queue", models.CharField(blank=True, default="", max_length=128)),
                ("priority", models.PositiveSmallIntegerField(default=0)),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("pending", "Pending"),
                            ("claimed", "Claimed"),
                            ("published", "Published"),
                            ("failed", "Failed"),
                        ],
                        db_index=True,
                        default="pending",
                        max_length=20,
                    ),
                ),
                ("attempts", models.PositiveIntegerField(default=0)),
                (
                    "claim_token",
                    models.CharField(blank=True, default="", max_length=64),
                ),
                (
                    "lease_expires_at",
                    models.DateTimeField(blank=True, db_index=True, null=True),
                ),
                (
                    "available_at",
                    models.DateTimeField(db_index=True, default=timezone.now),
                ),
                (
                    "broker_task_id",
                    models.CharField(blank=True, default="", max_length=255),
                ),
                ("last_error", models.TextField(blank=True, default="")),
                ("created_at", models.DateTimeField(auto_now_add=True, db_index=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("published_at", models.DateTimeField(blank=True, null=True)),
                (
                    "job",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="task_dispatches",
                        to="jobs.job",
                    ),
                ),
            ],
            options={
                "indexes": [
                    models.Index(
                        fields=["status", "available_at"],
                        name="jobs_dispatch_status_avail_idx",
                    ),
                    models.Index(
                        fields=["job", "stage", "status"],
                        name="jobs_dispatch_job_stage_idx",
                    ),
                ],
                "constraints": [
                    models.UniqueConstraint(
                        fields=("job", "stage", "generation"),
                        name="unique_task_dispatch_job_stage_generation",
                    ),
                ],
            },
        ),
    ]
