from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("jobs", "0028_page_ocr_review_provenance")]

    operations = [
        migrations.AddField(
            model_name="job",
            name="last_page_admission_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="pageresult",
            name="admitted_generation",
            field=models.PositiveIntegerField(blank=True, db_index=True, null=True),
        ),
    ]
