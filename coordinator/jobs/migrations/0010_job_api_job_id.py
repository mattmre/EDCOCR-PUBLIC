from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('jobs', '0009_n1b_compliance_controls'),
    ]

    operations = [
        migrations.AddField(
            model_name='job',
            name='api_job_id',
            field=models.CharField(blank=True, db_index=True, default='', max_length=64),
        ),
    ]
