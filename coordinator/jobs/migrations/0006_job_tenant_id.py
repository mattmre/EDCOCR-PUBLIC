# Add tenant_id to Job model for tenant-scoped Prometheus metrics

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('jobs', '0005_extraction_models'),
    ]

    operations = [
        migrations.AddField(
            model_name='job',
            name='tenant_id',
            field=models.CharField(
                blank=True, db_index=True, default='', max_length=128
            ),
        ),
    ]
