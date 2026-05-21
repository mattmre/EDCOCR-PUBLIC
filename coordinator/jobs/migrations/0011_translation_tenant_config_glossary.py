"""Add TranslationTenantConfig and GlossaryEntry models (Plan B Wave M2)."""

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('jobs', '0010_job_api_job_id'),
    ]

    operations = [
        migrations.CreateModel(
            name='TranslationTenantConfig',
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
                    'tenant_id',
                    models.CharField(
                        db_index=True,
                        help_text='Tenant identifier; must be unique.',
                        max_length=128,
                        unique=True,
                    ),
                ),
                (
                    'target_languages',
                    models.JSONField(
                        blank=True,
                        default=list,
                        help_text="Default target BCP-47 codes, e.g. ['en', 'fr'].",
                    ),
                ),
                (
                    'preferred_engines',
                    models.JSONField(
                        blank=True,
                        default=list,
                        help_text=(
                            "Engine ids in priority order, e.g. "
                            "['opus_mt', 'nllb_200']."
                        ),
                    ),
                ),
                (
                    'allow_nc_licensed',
                    models.BooleanField(
                        default=False,
                        help_text=(
                            'When True, NC-licensed engines (e.g. NLLB '
                            'CC-BY-NC-4.0) are eligible.'
                        ),
                    ),
                ),
                (
                    'require_certified',
                    models.BooleanField(
                        default=False,
                        help_text=(
                            'Tenant policy flag: requires translations to be '
                            "reviewed before any 'certified' attestation can "
                            'be applied.  This does NOT bypass the '
                            'certified=False enforcement at the sidecar '
                            'write path.'
                        ),
                    ),
                ),
                (
                    'default_quality_tier',
                    models.CharField(
                        blank=True,
                        default='standard',
                        help_text='One of: draft, standard, legal.',
                        max_length=20,
                    ),
                ),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
            options={
                'ordering': ['tenant_id'],
            },
        ),
        migrations.CreateModel(
            name='GlossaryEntry',
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
                    'tenant_id',
                    models.CharField(
                        db_index=True,
                        help_text=(
                            'Tenant identifier (matches '
                            'TranslationTenantConfig.tenant_id).'
                        ),
                        max_length=128,
                    ),
                ),
                ('source_term', models.CharField(max_length=500)),
                ('target_term', models.CharField(max_length=500)),
                (
                    'source_lang',
                    models.CharField(
                        help_text='BCP-47 code of source.',
                        max_length=10,
                    ),
                ),
                (
                    'target_lang',
                    models.CharField(
                        help_text='BCP-47 code of target.',
                        max_length=10,
                    ),
                ),
                ('case_sensitive', models.BooleanField(default=False)),
                (
                    'is_regex',
                    models.BooleanField(
                        default=False,
                        help_text=(
                            'When True, source_term is interpreted as a '
                            'Python regex.'
                        ),
                    ),
                ),
                (
                    'priority',
                    models.IntegerField(
                        default=100,
                        help_text='Lower priority is applied first (10 < 100).',
                    ),
                ),
                (
                    'notes',
                    models.TextField(blank=True, default='', null=True),
                ),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
            options={
                'ordering': ['tenant_id', 'priority', 'id'],
            },
        ),
        migrations.AddIndex(
            model_name='glossaryentry',
            index=models.Index(
                fields=['tenant_id'],
                name='jobs_glossa_tenant__4772ca_idx',
            ),
        ),
        migrations.AddIndex(
            model_name='glossaryentry',
            index=models.Index(
                fields=['tenant_id', 'source_lang', 'target_lang'],
                name='jobs_glossa_tenant__9acc5e_idx',
            ),
        ),
        migrations.AddConstraint(
            model_name='glossaryentry',
            constraint=models.UniqueConstraint(
                condition=models.Q(('is_regex', False)),
                fields=(
                    'tenant_id',
                    'source_term',
                    'source_lang',
                    'target_lang',
                ),
                name='unique_literal_glossary_entry',
            ),
        ),
    ]
