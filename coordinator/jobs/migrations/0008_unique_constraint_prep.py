"""Replace deprecated unique_together with UniqueConstraint.

Django 5.1 deprecated unique_together; Django 6.0 will remove it.
This migration converts both PageResult and DocumentChunk to use
models.UniqueConstraint for forward-compatibility.
"""

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('jobs', '0007_apikeyrecord'),
    ]

    operations = [
        # --- PageResult: unique_together -> UniqueConstraint ---
        migrations.AlterUniqueTogether(
            name='pageresult',
            unique_together=set(),
        ),
        migrations.AddConstraint(
            model_name='pageresult',
            constraint=models.UniqueConstraint(
                fields=['job', 'page_num'],
                name='unique_job_page_num',
            ),
        ),
        # --- DocumentChunk: unique_together -> UniqueConstraint ---
        migrations.AlterUniqueTogether(
            name='documentchunk',
            unique_together=set(),
        ),
        migrations.AddConstraint(
            model_name='documentchunk',
            constraint=models.UniqueConstraint(
                fields=['job', 'page_number', 'chunk_index'],
                name='unique_chunk',
            ),
        ),
    ]
