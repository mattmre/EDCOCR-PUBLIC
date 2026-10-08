"""S2-04: residency_region on tenant config + batch certified delivery gate.

Adds:
- TranslationTenantConfig.residency_region — pin for cloud engine rejection
- BatchTranslationJob.certified — promotion flag for require_certified delivery
"""

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("jobs", "0016_jobs_child_rls_postgres"),
    ]

    operations = [
        migrations.AddField(
            model_name="translationtenantconfig",
            name="residency_region",
            field=models.CharField(
                blank=True,
                default="",
                help_text=(
                    "Optional data-residency pin (e.g. 'eu-west-1', 'eu'). When set, "
                    "cloud translation engines are rejected with RESIDENCY_VIOLATION "
                    "until region-affinity cloud adapters exist (S2-04)."
                ),
                max_length=64,
            ),
        ),
        migrations.AddField(
            model_name="batchtranslationjob",
            name="certified",
            field=models.BooleanField(
                default=False,
                help_text=(
                    "True only after review certify with real custody linkage. "
                    "When tenant require_certified=True, result delivery is blocked "
                    "until this flag is True (S2-04)."
                ),
            ),
        ),
    ]
