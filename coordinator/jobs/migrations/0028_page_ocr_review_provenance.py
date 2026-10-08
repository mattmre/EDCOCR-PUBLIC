from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("jobs", "0027_merge_service_api_admission")]

    operations = [
        migrations.AlterField(
            model_name="job",
            name="detected_language",
            field=models.CharField(blank=True, default="und", max_length=20),
        ),
        migrations.AddField(
            model_name="pageresult",
            name="needs_review",
            field=models.BooleanField(default=True),
        ),
        migrations.AddField(
            model_name="pageresult",
            name="ocr_provenance",
            field=models.JSONField(blank=True, default=dict),
        ),
    ]
