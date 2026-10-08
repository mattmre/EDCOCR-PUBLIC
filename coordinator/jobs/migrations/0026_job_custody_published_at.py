from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("jobs", "0025_presigned_capability_generation"),
    ]

    operations = [
        migrations.AddField(
            model_name="job",
            name="custody_published_at",
            field=models.DateTimeField(
                blank=True,
                db_index=True,
                help_text=(
                    "Last terminal custody chain durably published to job storage."
                ),
                null=True,
            ),
        ),
    ]
