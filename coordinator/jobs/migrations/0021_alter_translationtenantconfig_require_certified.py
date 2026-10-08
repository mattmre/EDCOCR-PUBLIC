"""Record the current certification-policy help text in migration state."""

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("jobs", "0020_task_dispatch_outbox"),
    ]

    operations = [
        migrations.AlterField(
            model_name="translationtenantconfig",
            name="require_certified",
            field=models.BooleanField(
                default=False,
                help_text=(
                    "Tenant policy flag: requires translations to be reviewed "
                    "before any 'certified' attestation can be applied.  This "
                    "does NOT bypass the certified=False enforcement at the "
                    "sidecar write path. When True, delivery of uncertified "
                    "batch/job results is blocked (S2-04)."
                ),
            ),
        ),
    ]
