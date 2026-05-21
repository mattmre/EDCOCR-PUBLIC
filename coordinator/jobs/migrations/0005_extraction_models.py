# Phase 10: Persistent Intelligence Platform — extraction models

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('jobs', '0004_piientity'),
    ]

    operations = [
        migrations.CreateModel(
            name='ExtractedEntity',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('page_number', models.IntegerField()),
                ('entity_type', models.CharField(db_index=True, help_text='Entity type (e.g., PERSON, DATE, AMOUNT, CASE_NUMBER)', max_length=100)),
                ('entity_text', models.TextField()),
                ('confidence', models.FloatField(default=0.0)),
                ('bbox_x1', models.FloatField(blank=True, null=True)),
                ('bbox_y1', models.FloatField(blank=True, null=True)),
                ('bbox_x2', models.FloatField(blank=True, null=True)),
                ('bbox_y2', models.FloatField(blank=True, null=True)),
                ('source_module', models.CharField(help_text='Extraction source (ner, extraction, pii, barcode, etc.)', max_length=50)),
                ('metadata_json', models.JSONField(blank=True, default=dict)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('job', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='extracted_entities', to='jobs.job')),
            ],
            options={
                'ordering': ['job', 'page_number', 'entity_type'],
                'indexes': [
                    models.Index(fields=['entity_type', 'entity_text'], name='jobs_extrac_entity__bab32e_idx'),
                    models.Index(fields=['job', 'page_number'], name='jobs_extrac_job_id_e8f6a4_idx'),
                ],
            },
        ),
        migrations.CreateModel(
            name='ExtractedFormValue',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('page_number', models.IntegerField()),
                ('field_key', models.CharField(max_length=255)),
                ('field_value', models.TextField()),
                ('confidence', models.FloatField(default=0.0)),
                ('source_module', models.CharField(help_text='Extraction source (extraction, form_kv, etc.)', max_length=50)),
                ('metadata_json', models.JSONField(blank=True, default=dict)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('job', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='form_values', to='jobs.job')),
            ],
            options={
                'ordering': ['job', 'page_number', 'field_key'],
                'indexes': [
                    models.Index(fields=['field_key'], name='jobs_extrac_field_k_c4d1a7_idx'),
                    models.Index(fields=['job', 'page_number'], name='jobs_extrac_job_id_1a73b2_idx'),
                ],
            },
        ),
        migrations.CreateModel(
            name='DocumentChunk',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('page_number', models.IntegerField()),
                ('chunk_index', models.IntegerField()),
                ('chunk_text', models.TextField()),
                ('embedding_json', models.JSONField(blank=True, null=True)),
                ('embedding_model', models.CharField(blank=True, default='', max_length=100)),
                ('metadata_json', models.JSONField(blank=True, default=dict)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('job', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='chunks', to='jobs.job')),
            ],
            options={
                'ordering': ['job', 'page_number', 'chunk_index'],
                'unique_together': {('job', 'page_number', 'chunk_index')},
                'indexes': [
                    models.Index(fields=['job', 'page_number'], name='jobs_docume_job_id_a3f2c5_idx'),
                ],
            },
        ),
    ]
