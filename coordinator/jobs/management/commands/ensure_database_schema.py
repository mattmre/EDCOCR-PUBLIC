"""Create and validate the coordinator's isolated PostgreSQL schema."""

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import connection, transaction


class Command(BaseCommand):
    help = (
        "Create DATABASE_SCHEMA when absent and fail if it is owned by a "
        "different PostgreSQL role. Run before migrate."
    )

    def handle(self, *args, **options):
        schema = settings.DATABASE_SCHEMA
        if schema == "public":
            raise CommandError(
                "DATABASE_SCHEMA=public is not an isolated provider schema"
            )

        quoted_schema = connection.ops.quote_name(schema)
        with transaction.atomic(), connection.cursor() as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", [schema])
            cursor.execute(
                "SELECT pg_get_userbyid(nspowner) "
                "FROM pg_namespace WHERE nspname = %s",
                [schema],
            )
            row = cursor.fetchone()
            if row is None:
                cursor.execute(
                    f"CREATE SCHEMA {quoted_schema} AUTHORIZATION CURRENT_USER"
                )
            else:
                cursor.execute("SELECT CURRENT_USER")
                current_user = cursor.fetchone()[0]
                if row[0] != current_user:
                    raise CommandError(
                        f"Database schema {schema!r} is owned by another role"
                    )

            cursor.execute("SELECT current_schema()")
            selected = cursor.fetchone()[0]
            if selected != schema:
                raise CommandError(
                    f"Database search_path selected {selected!r}, expected {schema!r}"
                )

        self.stdout.write(self.style.SUCCESS(f"Database schema {schema!r} is ready"))
