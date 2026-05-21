"""Management command to audit API key access patterns.

Generates an access report for ApiKeyRecord entries over the last N days.
Groups by key_id showing total requests, last access time, and permissions.

Usage:
    python manage.py audit_api_access              # Last 30 days (default)
    python manage.py audit_api_access --days 7     # Last 7 days
    python manage.py audit_api_access --output json # Machine-readable JSON
"""

import json
import logging

from django.core.management.base import BaseCommand
from django.utils import timezone

from jobs.models import ApiKeyRecord

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Generate API key access audit report for the last N days"

    def add_arguments(self, parser):
        parser.add_argument(
            "--days",
            type=int,
            default=30,
            help="Report period in days (default: 30)",
        )
        parser.add_argument(
            "--output",
            choices=["text", "json"],
            default="text",
            help="Output format (default: text)",
        )

    def handle(self, *args, **options):
        days = options["days"]
        output_format = options["output"]

        cutoff = timezone.now() - timezone.timedelta(days=days)

        # Get all keys that were used within the period, plus inactive keys
        all_keys = list(ApiKeyRecord.objects.all())

        # Build report data
        report = {
            "period_days": days,
            "cutoff": cutoff.isoformat(),
            "generated_at": timezone.now().isoformat(),
            "total_keys": len(all_keys),
            "active_keys": sum(1 for k in all_keys if k.is_active),
            "revoked_keys": sum(1 for k in all_keys if not k.is_active),
            "keys": [],
        }

        for key in all_keys:
            used_in_period = (
                key.last_used_at is not None and key.last_used_at >= cutoff
            )
            key_data = {
                "key_id": key.key_id[:12] + "...",
                "description": key.description,
                "is_active": key.is_active,
                "use_count": key.use_count,
                "last_used_at": (
                    key.last_used_at.isoformat() if key.last_used_at else None
                ),
                "created_at": (
                    key.created_at.isoformat() if key.created_at else None
                ),
                "permissions": key.permissions,
                "used_in_period": used_in_period,
            }
            report["keys"].append(key_data)

        # Keys used within the period
        used_in_period = [k for k in report["keys"] if k["used_in_period"]]
        unused_in_period = [k for k in report["keys"] if not k["used_in_period"]]
        report["keys_used_in_period"] = len(used_in_period)
        report["keys_unused_in_period"] = len(unused_in_period)

        if output_format == "json":
            self.stdout.write(json.dumps(report, indent=2))
            return

        # Text output
        self.stdout.write("")
        self.stdout.write("=" * 70)
        self.stdout.write(f"  API Access Audit Report (last {days} days)")
        self.stdout.write("=" * 70)
        self.stdout.write("")
        self.stdout.write(f"  Total keys:         {report['total_keys']}")
        self.stdout.write(f"  Active:             {report['active_keys']}")
        self.stdout.write(f"  Revoked:            {report['revoked_keys']}")
        self.stdout.write(f"  Used in period:     {report['keys_used_in_period']}")
        self.stdout.write(f"  Unused in period:   {report['keys_unused_in_period']}")
        self.stdout.write("")

        if report["keys"]:
            self.stdout.write(
                f"  {'Key ID':<16s} {'Description':<20s} {'Active':<8s} "
                f"{'Uses':<8s} {'Last Used':<22s} {'In Period':<10s}"
            )
            self.stdout.write(
                f"  {'-' * 16} {'-' * 20} {'-' * 8} "
                f"{'-' * 8} {'-' * 22} {'-' * 10}"
            )

            for key_data in report["keys"]:
                desc = key_data["description"][:17] + "..." if len(key_data["description"]) > 20 else key_data["description"]
                active = "Yes" if key_data["is_active"] else "No"
                last_used = key_data["last_used_at"][:19] if key_data["last_used_at"] else "never"
                in_period = "Yes" if key_data["used_in_period"] else "No"

                self.stdout.write(
                    f"  {key_data['key_id']:<16s} {desc:<20s} {active:<8s} "
                    f"{str(key_data['use_count']):<8s} {last_used:<22s} {in_period:<10s}"
                )

        # Warnings
        self.stdout.write("")
        revoked_active = [
            k for k in report["keys"]
            if not k["is_active"] and k["used_in_period"]
        ]
        if revoked_active:
            self.stdout.write(
                self.style.WARNING(
                    f"  WARNING: {len(revoked_active)} revoked key(s) show "
                    f"activity in the audit period"
                )
            )

        unused_active = [
            k for k in report["keys"]
            if k["is_active"] and not k["used_in_period"]
        ]
        if unused_active:
            self.stdout.write(
                self.style.WARNING(
                    f"  WARNING: {len(unused_active)} active key(s) have not "
                    f"been used in the last {days} days -- consider revoking"
                )
            )

        self.stdout.write("")
        self.stdout.write("=" * 70)
