from django.db import migrations, models
from django.db.models import F


def backfill_dispatch_generation(apps, schema_editor):
    """Bind every pre-existing capability to its owning execution generation."""
    if schema_editor.connection.vendor == "postgresql":
        with schema_editor.connection.cursor() as cursor:
            cursor.execute("SET LOCAL app.control_plane = 'on'")

    Capability = apps.get_model("jobs", "PresignedTaskCapability")
    pending = []
    for capability in (
        Capability.objects.select_related("job")
        .order_by("capability_id")
        .iterator(chunk_size=2_000)
    ):
        capability.dispatch_generation = int(capability.job.dispatch_generation or 0)
        pending.append(capability)
        if len(pending) >= 2_000:
            Capability.objects.bulk_update(pending, ["dispatch_generation"])
            pending.clear()
    if pending:
        Capability.objects.bulk_update(pending, ["dispatch_generation"])

    if Capability.objects.exclude(
        dispatch_generation=F("job__dispatch_generation")
    ).exists():
        raise RuntimeError("Presigned capability generation backfill left stale rows")


class Migration(migrations.Migration):
    dependencies = [
        ("jobs", "0024_custody_chain_identity"),
    ]

    operations = [
        migrations.AddField(
            model_name="presignedtaskcapability",
            name="dispatch_generation",
            field=models.PositiveIntegerField(
                default=0,
                help_text=(
                    "Job dispatch generation authorized when the capability was issued."
                ),
            ),
        ),
        migrations.RunPython(
            backfill_dispatch_generation,
            migrations.RunPython.noop,
        ),
    ]
