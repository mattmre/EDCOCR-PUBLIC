"""Bound outbox retries and complete tenant RLS coverage for new tables.

SQLite remains a schema-only/no-RLS test backend. PostgreSQL receives FORCE
RLS policies for every job-child or direct-tenant table added after the
original 0015/0016 policy migrations.
"""

from django.db import migrations, models

import jobs.webhook_secret_field

_JOB_CHILD_TABLES = (
    "jobs_taskdispatchoutbox",
    "jobs_presignedtaskcapability",
    "jobs_documentchunk",
)

_DIRECT_TENANT_TABLES = (
    "jobs_translationtenantconfig",
    "jobs_glossaryentry",
    "jobs_batchtranslationjob",
)


def _require_table_sql(table: str) -> list[str]:
    return [
        f"  IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = '{table}') THEN",
        f"    RAISE EXCEPTION 'RLS migration requires {table}; refusing an unprotected database';",
        "  END IF;",
    ]


def _forward_sql() -> str:
    parts = [
        "DO $$",
        "BEGIN",
        "  IF current_setting('server_version_num')::int < 90500 THEN",
        "    RETURN;",
        "  END IF;",
    ]

    for table in _JOB_CHILD_TABLES:
        policy = f"{table}_tenant_isolation"
        parts.extend(_require_table_sql(table))
        parts.extend(
            [
                f"  ALTER TABLE {table} ENABLE ROW LEVEL SECURITY;",
                f"  ALTER TABLE {table} FORCE ROW LEVEL SECURITY;",
                f"  DROP POLICY IF EXISTS {policy} ON {table};",
                f"  CREATE POLICY {policy} ON {table}",
                "    USING (",
                "      current_setting('app.control_plane', true) = 'on'",
                "      OR (",
                "      current_setting('app.tenant_id', true) IS NOT NULL",
                "      AND current_setting('app.tenant_id', true) <> ''",
                "      AND EXISTS (",
                "        SELECT 1 FROM jobs_job j",
                f"        WHERE j.job_id = {table}.job_id",
                "          AND j.tenant_id = current_setting('app.tenant_id', true)",
                "      )",
                "      )",
                "    );",
            ]
        )

    for table in _DIRECT_TENANT_TABLES:
        policy = f"{table}_tenant_isolation"
        parts.extend(_require_table_sql(table))
        parts.extend(
            [
                f"  ALTER TABLE {table} ENABLE ROW LEVEL SECURITY;",
                f"  ALTER TABLE {table} FORCE ROW LEVEL SECURITY;",
                f"  DROP POLICY IF EXISTS {policy} ON {table};",
                f"  CREATE POLICY {policy} ON {table}",
                "    USING (",
                "      current_setting('app.control_plane', true) = 'on'",
                "      OR (",
                "      current_setting('app.tenant_id', true) IS NOT NULL",
                "      AND current_setting('app.tenant_id', true) <> ''",
                f"      AND {table}.tenant_id = current_setting('app.tenant_id', true)",
                "      )",
                "    );",
            ]
        )

    table = "jobs_batchtranslationinput"
    policy = f"{table}_tenant_isolation"
    parts.extend(_require_table_sql(table))
    parts.extend(
        [
            f"  ALTER TABLE {table} ENABLE ROW LEVEL SECURITY;",
            f"  ALTER TABLE {table} FORCE ROW LEVEL SECURITY;",
            f"  DROP POLICY IF EXISTS {policy} ON {table};",
            f"  CREATE POLICY {policy} ON {table}",
            "    USING (",
            "      current_setting('app.control_plane', true) = 'on'",
            "      OR (",
            "      current_setting('app.tenant_id', true) IS NOT NULL",
            "      AND current_setting('app.tenant_id', true) <> ''",
            "      AND EXISTS (",
            "        SELECT 1 FROM jobs_batchtranslationjob b",
            f"        WHERE b.batch_id = {table}.batch_id",
            "          AND b.tenant_id = current_setting('app.tenant_id', true)",
            "      )",
            "      )",
            "    );",
        ]
    )

    table = "jobs_custodyevent"
    policy = f"{table}_tenant_isolation"
    parts.extend(_require_table_sql(table))
    parts.extend(
        [
            f"  DROP POLICY IF EXISTS {policy} ON {table};",
            f"  CREATE POLICY {policy} ON {table}",
            "    USING (",
            "      current_setting('app.control_plane', true) = 'on'",
            "      OR (",
            "      current_setting('app.tenant_id', true) IS NOT NULL",
            "      AND current_setting('app.tenant_id', true) <> ''",
            "      AND (",
            f"        {table}.tenant_id = current_setting('app.tenant_id', true)",
            "        OR EXISTS (",
            "          SELECT 1 FROM jobs_job j",
            f"          WHERE j.job_id = {table}.job_id",
            "            AND j.tenant_id = current_setting('app.tenant_id', true)",
            "        )",
            "      )",
            "      )",
            "    );",
            "END $$;",
        ]
    )

    # The durable relay filters parent lifecycle state while operating across
    # tenants.  A separate policy keeps ordinary tenant isolation intact and
    # makes that control-plane access explicit instead of depending on an
    # accidental BYPASSRLS deployment role.
    parts[-1:-1] = [
        "  DROP POLICY IF EXISTS jobs_job_control_plane ON jobs_job;",
        "  CREATE POLICY jobs_job_control_plane ON jobs_job",
        "    USING (current_setting('app.control_plane', true) = 'on');",
    ]
    return "\n".join(parts)


def _reverse_sql() -> str:
    tables = (*_JOB_CHILD_TABLES, *_DIRECT_TENANT_TABLES, "jobs_batchtranslationinput")
    parts = ["DO $$", "BEGIN"]
    for table in tables:
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
    parts.extend(
        [
            "  BEGIN",
            "    DROP POLICY IF EXISTS jobs_job_control_plane ON jobs_job;",
            "  EXCEPTION WHEN undefined_table THEN",
            "    NULL;",
            "  END;",
            "  BEGIN",
            "    DROP POLICY IF EXISTS jobs_custodyevent_control_plane ON jobs_custodyevent;",
            "  EXCEPTION WHEN undefined_table THEN",
            "    NULL;",
            "  END;",
            "  BEGIN",
            "    DROP POLICY IF EXISTS jobs_custodyevent_tenant_isolation ON jobs_custodyevent;",
            "    CREATE POLICY jobs_custodyevent_tenant_isolation ON jobs_custodyevent",
            "      USING (",
            "        current_setting('app.tenant_id', true) IS NOT NULL",
            "        AND current_setting('app.tenant_id', true) <> ''",
            "        AND EXISTS (",
            "          SELECT 1 FROM jobs_job j",
            "          WHERE j.job_id = jobs_custodyevent.job_id",
            "            AND j.tenant_id = current_setting('app.tenant_id', true)",
            "        )",
            "      );",
            "  EXCEPTION WHEN undefined_table THEN",
            "    NULL;",
            "  END;",
            "END $$;",
        ]
    )
    return "\n".join(parts)


def _apply_rls(apps, schema_editor):
    if schema_editor.connection.vendor != "postgresql":
        return
    with schema_editor.connection.cursor() as cursor:
        cursor.execute(_forward_sql())


def _prepare_control_plane_data_migration(apps, schema_editor):
    """Make forced-RLS historical rows visible to this atomic migration."""
    del apps
    if schema_editor.connection.vendor != "postgresql":
        return
    with schema_editor.connection.cursor() as cursor:
        cursor.execute(
            """
            DROP POLICY IF EXISTS jobs_job_control_plane ON jobs_job;
            CREATE POLICY jobs_job_control_plane ON jobs_job
              USING (current_setting('app.control_plane', true) = 'on');
            DROP POLICY IF EXISTS jobs_custodyevent_control_plane ON jobs_custodyevent;
            CREATE POLICY jobs_custodyevent_control_plane ON jobs_custodyevent
              USING (current_setting('app.control_plane', true) = 'on');
            SET LOCAL app.control_plane = 'on';
            """
        )


def _backfill_custody_tenants(apps, schema_editor):
    CustodyEvent = apps.get_model("jobs", "CustodyEvent")
    pending = []
    rows = CustodyEvent.objects.filter(tenant_id="", job__isnull=False).select_related(
        "job"
    )
    for event in rows.iterator(chunk_size=1_000):
        event.tenant_id = str(getattr(event.job, "tenant_id", "") or "").strip()
        pending.append(event)
        if len(pending) >= 1_000:
            CustodyEvent.objects.bulk_update(pending, ["tenant_id"], batch_size=1_000)
            pending.clear()
    if pending:
        CustodyEvent.objects.bulk_update(pending, ["tenant_id"], batch_size=1_000)


def _encrypt_webhook_secrets(apps, schema_editor):
    Job = apps.get_model("jobs", "Job")
    from jobs.webhook_secret_field import (
        encrypt_webhook_secret,
        is_encrypted_webhook_secret,
    )

    pending = []
    for job in Job.objects.exclude(webhook_secret="").iterator(chunk_size=500):
        raw = str(job.webhook_secret or "")
        if is_encrypted_webhook_secret(raw):
            continue
        job.webhook_secret = encrypt_webhook_secret(raw)
        pending.append(job)
        if len(pending) >= 500:
            Job.objects.bulk_update(pending, ["webhook_secret"], batch_size=500)
            pending.clear()
    if pending:
        Job.objects.bulk_update(pending, ["webhook_secret"], batch_size=500)


def _reverse_rls(apps, schema_editor):
    if schema_editor.connection.vendor != "postgresql":
        return
    with schema_editor.connection.cursor() as cursor:
        cursor.execute(_reverse_sql())


class Migration(migrations.Migration):
    dependencies = [("jobs", "0021_alter_translationtenantconfig_require_certified")]

    operations = [
        migrations.AddField(
            model_name="custodyevent",
            name="tenant_id",
            field=models.CharField(
                blank=True,
                db_index=True,
                default="",
                help_text=(
                    "Persisted tenant binding for jobless control-plane events; "
                    "job-backed events inherit their parent tenant."
                ),
                max_length=128,
            ),
        ),
        migrations.AddField(
            model_name="taskdispatchoutbox",
            name="dead_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="taskdispatchoutbox",
            name="max_attempts",
            field=models.PositiveSmallIntegerField(
                default=12,
                help_text="Bounded broker publish attempts before dead-lettering.",
            ),
        ),
        migrations.AlterField(
            model_name="taskdispatchoutbox",
            name="status",
            field=models.CharField(
                choices=[
                    ("pending", "Pending"),
                    ("claimed", "Claimed"),
                    ("published", "Published"),
                    ("failed", "Failed"),
                    ("dead", "Dead letter"),
                ],
                db_index=True,
                default="pending",
                max_length=20,
            ),
        ),
        migrations.AlterField(
            model_name="job",
            name="webhook_secret",
            field=models.TextField(blank=True, default=""),
        ),
        migrations.RunPython(
            _prepare_control_plane_data_migration, migrations.RunPython.noop
        ),
        migrations.RunPython(_encrypt_webhook_secrets, migrations.RunPython.noop),
        migrations.AlterField(
            model_name="job",
            name="webhook_secret",
            field=jobs.webhook_secret_field.WebhookSecretField(blank=True, default=""),
        ),
        migrations.RunPython(_backfill_custody_tenants, migrations.RunPython.noop),
        migrations.RunPython(_apply_rls, reverse_code=_reverse_rls),
    ]
