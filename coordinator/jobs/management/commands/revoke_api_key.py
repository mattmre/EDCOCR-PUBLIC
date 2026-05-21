"""Management command to revoke an API key.

Sets the ApiKeyRecord.is_active flag to False and emits a CustodyEvent
for the forensic audit trail.

Usage:
    python manage.py revoke_api_key --key-id abc123...  --confirm
    python manage.py revoke_api_key --key-id abc123...           # Preview only
"""

import getpass
import logging

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from jobs.models import ApiKeyRecord, CustodyEvent, Job

logger = logging.getLogger(__name__)


def _record_revocation_custody_event(key_record, operator):
    """Create a CustodyEvent recording an API key revocation.

    Anchors to the first available Job, or logs if none exist (same
    pattern as cleanup_old_jobs.py / purge_temp_files.py).
    """
    try:
        first_job = Job.objects.first()
        if first_job is not None:
            CustodyEvent.objects.create(
                document_id=f"api-key-revoke-{timezone.now().isoformat()}",
                job=first_job,
                event_type="api_key_revoked",
                data={
                    "action": "revoke_api_key",
                    "key_id": key_record.key_id,
                    "description": key_record.description,
                    "use_count_at_revocation": key_record.use_count,
                    "reason": "manual_revocation",
                    "operator": operator,
                },
            )
        else:
            logger.info(
                "No Job available to anchor custody event -- logging instead: "
                "api_key_revoked key_id=%s operator=%s",
                key_record.key_id[:12],
                operator,
            )
    except Exception:
        logger.exception("Failed to record API key revocation custody event")


class Command(BaseCommand):
    help = "Revoke an API key (set is_active=False) with audit trail"

    def add_arguments(self, parser):
        parser.add_argument(
            "--key-id",
            required=True,
            help="The key_id (or prefix) of the API key to revoke",
        )
        parser.add_argument(
            "--confirm",
            action="store_true",
            help="Actually perform the revocation (without this flag, preview only)",
        )

    def handle(self, *args, **options):
        key_id = options["key_id"]
        confirm = options["confirm"]

        # Look up by exact match first, then by prefix
        try:
            key_record = ApiKeyRecord.objects.get(key_id=key_id)
        except ApiKeyRecord.DoesNotExist:
            # Try prefix match
            candidates = list(
                ApiKeyRecord.objects.filter(key_id__startswith=key_id)
            )
            if len(candidates) == 0:
                raise CommandError(f"No API key found matching '{key_id}'")
            if len(candidates) > 1:
                raise CommandError(
                    f"Ambiguous key_id prefix '{key_id}' matches "
                    f"{len(candidates)} keys. Provide a longer prefix."
                )
            key_record = candidates[0]

        # Display key info
        truncated = key_record.key_id[:12] + '...'
        self.stdout.write("")
        self.stdout.write(f"  Key ID:       {truncated}")
        self.stdout.write(f"  Full ID:      {key_record.key_id}")
        self.stdout.write(f"  Description:  {key_record.description or '(none)'}")
        self.stdout.write(f"  Active:       {'Yes' if key_record.is_active else 'No'}")
        self.stdout.write(f"  Use count:    {key_record.use_count}")
        self.stdout.write(f"  Created:      {key_record.created_at}")
        self.stdout.write(f"  Last used:    {key_record.last_used_at or 'never'}")
        self.stdout.write("")

        if not key_record.is_active:
            self.stdout.write(
                self.style.WARNING("  This key is already revoked.")
            )
            return

        if not confirm:
            self.stdout.write(
                self.style.WARNING(
                    "  DRY RUN: Add --confirm to actually revoke this key."
                )
            )
            return

        # Perform revocation
        key_record.is_active = False
        key_record.save()

        # Emit custody event
        operator = getpass.getuser()
        _record_revocation_custody_event(key_record, operator)

        self.stdout.write(
            self.style.SUCCESS(
                f"  API key {truncated} has been revoked."
            )
        )
