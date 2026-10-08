from django.db import migrations


class Migration(migrations.Migration):
    """Merge processing admission fields onto the closeout 0026 lineage."""

    dependencies = [
        ("jobs", "0022_job_service_api_admission"),
        ("jobs", "0026_job_custody_published_at"),
    ]

    operations = []
