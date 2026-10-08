"""Persist the fan-out dispatch guard used by extract_pages."""

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("jobs", "0017_translation_residency_require_certified_delivery"),
    ]

    operations = [
        migrations.AddField(
            model_name="job",
            name="fanout_dispatched",
            field=models.BooleanField(default=False),
        ),
    ]
