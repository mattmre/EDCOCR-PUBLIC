# Add ApiKeyRecord model for API key usage tracking and access review

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('jobs', '0006_job_tenant_id'),
    ]

    operations = [
        migrations.CreateModel(
            name='ApiKeyRecord',
            fields=[
                (
                    'id',
                    models.AutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name='ID',
                    ),
                ),
                (
                    'key_id',
                    models.CharField(
                        help_text='SHA-256 hash of the API key (never store raw keys)',
                        max_length=64,
                        unique=True,
                    ),
                ),
                (
                    'description',
                    models.CharField(blank=True, default='', max_length=255),
                ),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('last_used_at', models.DateTimeField(blank=True, null=True)),
                ('use_count', models.IntegerField(default=0)),
                ('is_active', models.BooleanField(db_index=True, default=True)),
                (
                    'permissions',
                    models.JSONField(
                        default=list,
                        help_text="List of permission strings, e.g. ['read', 'write', 'admin']",
                    ),
                ),
            ],
            options={
                'ordering': ['-created_at'],
            },
        ),
    ]
