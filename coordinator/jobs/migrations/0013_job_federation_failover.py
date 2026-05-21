"""Add federation failover tracking fields (Plan C Phase 1, item C5).

* ``Job.assigned_cluster`` -- which federation peer currently owns this
  job (empty string means local / not federated).
* ``Job.last_failure_reason`` -- machine-readable tag set by the failover
  engine and worker retry hooks.
"""

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('jobs', '0012_batch_translation'),
    ]

    operations = [
        migrations.AddField(
            model_name='job',
            name='assigned_cluster',
            field=models.CharField(
                blank=True,
                db_index=True,
                default='',
                max_length=128,
            ),
        ),
        migrations.AddField(
            model_name='job',
            name='last_failure_reason',
            field=models.CharField(
                blank=True,
                default='',
                max_length=64,
            ),
        ),
    ]
