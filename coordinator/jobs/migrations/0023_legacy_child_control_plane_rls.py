"""Authorize bounded fleet maintenance on legacy FORCE-RLS child tables."""

from django.db import migrations

_TABLES = (
    "jobs_pageresult",
    "jobs_piientity",
    "jobs_extractedentity",
    "jobs_extractedformvalue",
)


def _apply(apps, schema_editor):
    if schema_editor.connection.vendor != "postgresql":
        return
    with schema_editor.connection.cursor() as cursor:
        for table in _TABLES:
            policy = f"{table}_control_plane"
            cursor.execute(f"DROP POLICY IF EXISTS {policy} ON {table}")
            cursor.execute(
                f"""CREATE POLICY {policy} ON {table}
                    FOR ALL
                    USING (current_setting('app.control_plane', true) = 'on')
                    WITH CHECK (current_setting('app.control_plane', true) = 'on')"""
            )


def _reverse(apps, schema_editor):
    if schema_editor.connection.vendor != "postgresql":
        return
    with schema_editor.connection.cursor() as cursor:
        for table in _TABLES:
            cursor.execute(f"DROP POLICY IF EXISTS {table}_control_plane ON {table}")


class Migration(migrations.Migration):
    dependencies = [("jobs", "0022_outbox_deadletter_custody_tenant_rls")]

    operations = [migrations.RunPython(_apply, reverse_code=_reverse)]
