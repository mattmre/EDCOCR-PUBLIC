"""Postgres RLS for jobs.tenant_id (R4-RC-07/08).

On PostgreSQL, enables FORCE row-level security so sessions that set
``app.tenant_id`` only see matching rows. Application code remains the
primary pin; RLS is defense-in-depth for SQL-level access.

SQLite (tests) is a no-op — RLS is Postgres-only.
"""

from django.db import migrations

SQL_FORWARD = """
DO $$
BEGIN
  IF current_setting('server_version_num')::int >= 90500 THEN
    IF NOT EXISTS (
      SELECT 1 FROM information_schema.tables WHERE table_name = 'jobs_job'
    ) THEN
      RAISE EXCEPTION 'RLS migration requires jobs_job; refusing an unprotected database';
    END IF;
    ALTER TABLE jobs_job ENABLE ROW LEVEL SECURITY;
    ALTER TABLE jobs_job FORCE ROW LEVEL SECURITY;
    DROP POLICY IF EXISTS jobs_tenant_isolation ON jobs_job;
    CREATE POLICY jobs_tenant_isolation ON jobs_job
      USING (
        current_setting('app.tenant_id', true) IS NOT NULL
        AND current_setting('app.tenant_id', true) <> ''
        AND tenant_id = current_setting('app.tenant_id', true)
      );
  END IF;
END $$;
"""

SQL_REVERSE = """
DO $$
BEGIN
  BEGIN
    DROP POLICY IF EXISTS jobs_tenant_isolation ON jobs_job;
  EXCEPTION WHEN undefined_table THEN
    NULL;
  END;
  BEGIN
    ALTER TABLE jobs_job NO FORCE ROW LEVEL SECURITY;
    ALTER TABLE jobs_job DISABLE ROW LEVEL SECURITY;
  EXCEPTION WHEN undefined_table THEN
    NULL;
  END;
END $$;
"""


def _apply_rls(apps, schema_editor):
    if schema_editor.connection.vendor != "postgresql":
        return
    with schema_editor.connection.cursor() as cursor:
        cursor.execute(SQL_FORWARD)


def _reverse_rls(apps, schema_editor):
    if schema_editor.connection.vendor != "postgresql":
        return
    with schema_editor.connection.cursor() as cursor:
        cursor.execute(SQL_REVERSE)


class Migration(migrations.Migration):
    dependencies = [
        (
            "jobs",
            "0014_rename_jobs_docume_job_id_a3f2c5_idx_jobs_docume_job_id_2cb984_idx_and_more",
        ),
    ]

    operations = [
        migrations.RunPython(_apply_rls, reverse_code=_reverse_rls),
    ]
