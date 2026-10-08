from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("jobs", "0021_alter_translationtenantconfig_require_certified")]

    operations = [
        migrations.AddField(
            model_name="job",
            name="source_object_key",
            field=models.CharField(blank=True, default="", max_length=1024),
        ),
        migrations.AddField(
            model_name="job",
            name="source_size_bytes",
            field=models.BigIntegerField(default=0),
        ),
        migrations.AddField(
            model_name="job",
            name="submission_key_hash",
            field=models.CharField(blank=True, default="", max_length=64),
        ),
        migrations.AddField(
            model_name="job",
            name="submission_ready",
            field=models.BooleanField(db_index=True, default=False),
        ),
        migrations.AddConstraint(
            model_name="job",
            constraint=models.UniqueConstraint(
                condition=~models.Q(submission_key_hash=""),
                fields=("tenant_id", "submission_key_hash"),
                name="unique_job_tenant_submission_key",
            ),
        ),
    ]
