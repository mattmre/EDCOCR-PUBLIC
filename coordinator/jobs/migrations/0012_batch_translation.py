"""Add BatchTranslationJob and BatchTranslationInput models (Plan B Wave M2 -- B17)."""

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('jobs', '0011_translation_tenant_config_glossary'),
    ]

    operations = [
        migrations.CreateModel(
            name='BatchTranslationJob',
            fields=[
                (
                    'batch_id',
                    models.CharField(
                        primary_key=True,
                        serialize=False,
                        max_length=64,
                        help_text='UUID4 hex of the batch.',
                    ),
                ),
                (
                    'tenant_id',
                    models.CharField(
                        max_length=128,
                        db_index=True,
                        help_text='Tenant identifier.',
                    ),
                ),
                (
                    'source_lang',
                    models.CharField(
                        max_length=10,
                        help_text='BCP-47 source language.',
                    ),
                ),
                (
                    'target_lang',
                    models.CharField(
                        max_length=10,
                        help_text='BCP-47 target language.',
                    ),
                ),
                (
                    'status',
                    models.CharField(
                        max_length=20,
                        default='pending',
                        db_index=True,
                        help_text=(
                            'One of: pending, running, completed, failed, '
                            'cancelled.'
                        ),
                    ),
                ),
                (
                    'priority',
                    models.IntegerField(
                        default=0,
                        help_text='Celery priority 0..9 (lower = lower priority).',
                    ),
                ),
                (
                    'glossary_enabled',
                    models.BooleanField(default=True),
                ),
                (
                    'total_inputs',
                    models.IntegerField(default=0),
                ),
                (
                    'completed_inputs',
                    models.IntegerField(default=0),
                ),
                (
                    'failed_inputs',
                    models.IntegerField(default=0),
                ),
                (
                    'submitted_at',
                    models.DateTimeField(auto_now_add=True),
                ),
                (
                    'completed_at',
                    models.DateTimeField(blank=True, null=True),
                ),
            ],
            options={
                'ordering': ['-submitted_at'],
            },
        ),
        migrations.CreateModel(
            name='BatchTranslationInput',
            fields=[
                (
                    'id',
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name='ID',
                    ),
                ),
                (
                    'batch',
                    models.ForeignKey(
                        on_delete=models.CASCADE,
                        related_name='inputs',
                        to='jobs.batchtranslationjob',
                    ),
                ),
                (
                    'client_ref',
                    models.CharField(
                        max_length=128,
                        help_text='Tenant-supplied opaque id, unique per batch.',
                    ),
                ),
                (
                    'input_index',
                    models.IntegerField(
                        default=0,
                        help_text='Position in the original submitted list.',
                    ),
                ),
                ('text', models.TextField()),
                (
                    'status',
                    models.CharField(
                        max_length=20,
                        default='pending',
                        db_index=True,
                    ),
                ),
                (
                    'target_text',
                    models.TextField(blank=True, default=''),
                ),
                (
                    'engine_id',
                    models.CharField(
                        max_length=64,
                        blank=True,
                        default='',
                    ),
                ),
                (
                    'confidence',
                    models.FloatField(blank=True, null=True),
                ),
                (
                    'glossary_hits_json',
                    models.JSONField(blank=True, default=list),
                ),
                (
                    'error',
                    models.TextField(blank=True, default=''),
                ),
                (
                    'celery_task_id',
                    models.CharField(
                        max_length=128,
                        blank=True,
                        default='',
                    ),
                ),
                ('started_at', models.DateTimeField(blank=True, null=True)),
                ('completed_at', models.DateTimeField(blank=True, null=True)),
                (
                    'optional_metadata_json',
                    models.JSONField(blank=True, default=dict),
                ),
            ],
            options={
                'ordering': ['batch', 'input_index'],
            },
        ),
        migrations.AddIndex(
            model_name='batchtranslationinput',
            index=models.Index(
                fields=['batch', 'status'],
                name='jobs_btinp_batch_s_idx',
            ),
        ),
        migrations.AddConstraint(
            model_name='batchtranslationinput',
            constraint=models.UniqueConstraint(
                fields=('batch', 'client_ref'),
                name='unique_batch_client_ref',
            ),
        ),
    ]
