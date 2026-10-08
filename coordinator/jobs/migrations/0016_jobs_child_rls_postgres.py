"""Postgres RLS for job-child tables (C05 / R4-RC-07/08 residual).

FORCE RLS on PageResult, CustodyEvent, PiiEntity, ExtractedEntity, and
ExtractedFormValue using EXISTS join to ``jobs_job.tenant_id`` and the same
``app.tenant_id`` GUC as migration 0015.

SQLite (tests) is a no-op — RLS is Postgres-only.
"""

from django.db import migrations

# Tables with FK job_id → jobs_job.job_id.  Job uses ``job_id`` as its
# primary key, so there is no implicit ``jobs_job.id`` column to join on.
_CHILD_TABLES = (
    "jobs_pageresult",
    "jobs_custodyevent",
    "jobs_piientity",
    "jobs_extractedentity",
    "jobs_extractedformvalue",
)


def _forward_sql() -> str:
    parts = [
        "DO $$",
        "BEGIN",
        "  IF current_setting('server_version_num')::int < 90500 THEN",
        "    RETURN;",
        "  END IF;",
    ]
    for table in _CHILD_TABLES:
        policy = f"{table}_tenant_isolation"
        parts.extend(
            [
                f"  IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = '{table}') THEN",
                f"    RAISE EXCEPTION 'RLS migration requires {table}; refusing an unprotected database';",
                "  END IF;",
                f"  ALTER TABLE {table} ENABLE ROW LEVEL SECURITY;",
                f"  ALTER TABLE {table} FORCE ROW LEVEL SECURITY;",
                f"  DROP POLICY IF EXISTS {policy} ON {table};",
                f"  CREATE POLICY {policy} ON {table}",
                "    USING (",
                "      current_setting('app.tenant_id', true) IS NOT NULL",
                "      AND current_setting('app.tenant_id', true) <> ''",
                "      AND EXISTS (",
                "        SELECT 1 FROM jobs_job j",
                f"        WHERE j.job_id = {table}.job_id",
                "          AND j.tenant_id = current_setting('app.tenant_id', true)",
                "      )",
                "    );",
            ]
        )
    parts.extend(["END $$;"])
    return "\n".join(parts)


def _reverse_sql() -> str:
    parts = ["DO $$", "BEGIN"]
    for table in _CHILD_TABLES:
        policy = f"{table}_tenant_isolation"
        parts.extend(
            [
                "  BEGIN",
                f"    DROP POLICY IF EXISTS {policy} ON {table};",
                f"    ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY;",
                f"    ALTER TABLE {table} DISABLE ROW LEVEL SECURITY;",
                "  EXCEPTION WHEN undefined_table THEN",
                "    NULL;",
                "  END;",
            ]
        )
    parts.extend(["END $$;"])
    return "\n".join(parts)


def _apply_rls(apps, schema_editor):
    if schema_editor.connection.vendor != "postgresql":
        return
    with schema_editor.connection.cursor() as cursor:
        cursor.execute(_forward_sql())


def _reverse_rls(apps, schema_editor):
    if schema_editor.connection.vendor != "postgresql":
        return
    with schema_editor.connection.cursor() as cursor:
        cursor.execute(_reverse_sql())


class Migration(migrations.Migration):
    dependencies = [
        ("jobs", "0015_jobs_rls_postgres"),
    ]

    operations = [
        migrations.RunPython(_apply_rls, reverse_code=_reverse_rls),
    ]
