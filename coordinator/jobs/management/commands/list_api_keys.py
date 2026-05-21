"""Management command to list registered API keys.

Displays ApiKeyRecord entries with usage statistics in a formatted table.
Key IDs are truncated for display; full IDs are never printed.

Usage:
    python manage.py list_api_keys                # List all keys
    python manage.py list_api_keys --active-only  # Active keys only
"""

from django.core.management.base import BaseCommand

from jobs.models import ApiKeyRecord


class Command(BaseCommand):
    help = "List registered API keys with usage statistics"

    def add_arguments(self, parser):
        parser.add_argument(
            "--active-only",
            action="store_true",
            help="Show only active (non-revoked) keys",
        )

    def handle(self, *args, **options):
        active_only = options["active_only"]

        qs = ApiKeyRecord.objects.all()
        if active_only:
            qs = qs.filter(is_active=True)

        keys = list(qs)

        if not keys:
            label = "active " if active_only else ""
            self.stdout.write(f"No {label}API keys found.")
            return

        self.stdout.write("")
        self.stdout.write(
            f"  {'Key ID':<16s} {'Description':<25s} {'Active':<8s} "
            f"{'Uses':<8s} {'Last Used':<22s} {'Created':<22s}"
        )
        self.stdout.write(
            f"  {'-' * 16} {'-' * 25} {'-' * 8} "
            f"{'-' * 8} {'-' * 22} {'-' * 22}"
        )

        for key in keys:
            key_id = key.key_id[:12] + '...' if len(key.key_id) > 12 else key.key_id
            desc = (key.description[:22] + '...') if len(key.description) > 25 else key.description
            active = "Yes" if key.is_active else "No"
            uses = str(key.use_count)
            last_used = str(key.last_used_at)[:19] if key.last_used_at else "never"
            created = str(key.created_at)[:19] if key.created_at else "-"

            self.stdout.write(
                f"  {key_id:<16s} {desc:<25s} {active:<8s} "
                f"{uses:<8s} {last_used:<22s} {created:<22s}"
            )

        self.stdout.write("")
        total = len(keys)
        active_count = sum(1 for k in keys if k.is_active)
        self.stdout.write(
            f"  Total: {total} key(s), {active_count} active, "
            f"{total - active_count} revoked"
        )
        self.stdout.write("")
