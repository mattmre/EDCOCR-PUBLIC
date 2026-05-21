"""N1-B compliance controls: custody SET_NULL, encrypted fields, PII flag.

Changes:
- CustodyEvent.job: CASCADE -> SET_NULL with null=True, blank=True
  (custody events must survive job deletion for forensic audit trail)
- ExtractedEntity.entity_text: TextField -> EncryptedTextField
  (application-level encryption at rest for PII/PHI)
- ExtractedFormValue.field_value: TextField -> EncryptedTextField
  (application-level encryption at rest for PII/PHI)
- DocumentChunk.contains_pii: new BooleanField(default=False)
  (flag for selective future encryption of PII-bearing chunks)
"""

import django.db.models.deletion
from django.db import migrations, models

import jobs.encrypted_field


class Migration(migrations.Migration):

    dependencies = [
        ('jobs', '0008_unique_constraint_prep'),
    ]

    operations = [
        # CustodyEvent.job: CASCADE -> SET_NULL, allow null
        migrations.AlterField(
            model_name='custodyevent',
            name='job',
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name='custody_events',
                to='jobs.job',
            ),
        ),
        # ExtractedEntity.entity_text: TextField -> EncryptedTextField
        migrations.AlterField(
            model_name='extractedentity',
            name='entity_text',
            field=jobs.encrypted_field.EncryptedTextField(),
        ),
        # ExtractedFormValue.field_value: TextField -> EncryptedTextField
        migrations.AlterField(
            model_name='extractedformvalue',
            name='field_value',
            field=jobs.encrypted_field.EncryptedTextField(),
        ),
        # DocumentChunk.contains_pii: new BooleanField
        migrations.AddField(
            model_name='documentchunk',
            name='contains_pii',
            field=models.BooleanField(
                default=False,
                help_text='Flag for selective future encryption of PII-bearing chunks',
            ),
        ),
    ]
