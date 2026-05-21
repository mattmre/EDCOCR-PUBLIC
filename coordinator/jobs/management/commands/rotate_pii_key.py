"""Management command to rotate the PII encryption key.

Re-encrypts all PiiEntity.entity_value records from the old key to the new
key in batches.  Emits a CustodyEvent for forensic audit trail.

Usage:
    python manage.py rotate_pii_key --old-key <base64_key> --new-key <base64_key>
    python manage.py rotate_pii_key --old-key <key> --new-key <key> --dry-run
    python manage.py rotate_pii_key --old-key <key> --new-key <key> --batch-size 500
"""

import getpass
import logging
import time

from django.core.management.base import BaseCommand
from django.utils import timezone

from jobs.models import CustodyEvent, Job, PiiEntity
from jobs.pii_encryption import PiiEncryptor

logger = logging.getLogger(__name__)

_DEFAULT_BATCH_SIZE = 1000


class Command(BaseCommand):
    help = "Re-encrypt all PII entity values with a new encryption key"

    def add_arguments(self, parser):
        parser.add_argument(
            "--old-key",
            required=True,
            help="Current encryption key (URL-safe base64, 32 bytes)",
        )
        parser.add_argument(
            "--new-key",
            required=True,
            help="New encryption key (URL-safe base64, 32 bytes)",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=_DEFAULT_BATCH_SIZE,
            help=f"Records per batch (default: {_DEFAULT_BATCH_SIZE})",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Preview rotation without modifying records",
        )

    def handle(self, *args, **options):
        old_key = options["old_key"]
        new_key = options["new_key"]
        batch_size = options["batch_size"]
        dry_run = options["dry_run"]

        if old_key == new_key:
            self.stderr.write(self.style.WARNING("Old and new keys are identical. Nothing to do."))
            return

        # Validate both keys can create Fernet instances
        try:
            encryptor = PiiEncryptor(old_key)
            if not encryptor.enabled:
                self.stderr.write(self.style.ERROR("Old key is invalid or cryptography is not installed."))
                return
        except Exception as exc:
            self.stderr.write(self.style.ERROR(f"Invalid old key: {exc}"))
            return

        try:
            new_encryptor = PiiEncryptor(new_key)
            if not new_encryptor.enabled:
                self.stderr.write(self.style.ERROR("New key is invalid."))
                return
        except Exception as exc:
            self.stderr.write(self.style.ERROR(f"Invalid new key: {exc}"))
            return

        total = PiiEntity.objects.count()
        if total == 0:
            self.stdout.write("No PII entities to rotate.")
            return

        if dry_run:
            self.stdout.write(
                f"DRY RUN: Would re-encrypt {total} PII entity records "
                f"in batches of {batch_size}."
            )
            return

        # Record custody event before rotation
        self._record_custody_event(
            event_type="pii_key_rotation_started",
            data={
                "action": "rotate_pii_key",
                "total_records": total,
                "batch_size": batch_size,
                "operator": getpass.getuser(),
                "timestamp": timezone.now().isoformat(),
            },
        )

        rotated = 0
        failed = 0
        start_time = time.monotonic()

        # Process in batches using primary-key pagination
        last_pk = 0
        while True:
            batch = list(
                PiiEntity.objects.filter(pk__gt=last_pk)
                .order_by("pk")[:batch_size]
            )
            if not batch:
                break

            for entity in batch:
                last_pk = entity.pk
                raw_value = entity.entity_value
                if not raw_value:
                    continue

                try:
                    if PiiEncryptor.is_encrypted(raw_value):
                        # Re-encrypt: old key -> plaintext -> new key
                        new_value = encryptor.rotate_key(old_key, new_key, raw_value)
                    else:
                        # Plaintext record: encrypt with new key
                        new_value = new_encryptor.encrypt(raw_value)

                    # Direct update to bypass EncryptedTextField auto-encrypt
                    PiiEntity.objects.filter(pk=entity.pk).update(
                        entity_value=new_value
                    )
                    rotated += 1
                except Exception:
                    failed += 1
                    logger.exception(
                        "Failed to rotate key for PiiEntity pk=%s", entity.pk
                    )

            self.stdout.write(
                f"  Progress: {rotated + failed}/{total} "
                f"({rotated} rotated, {failed} failed)"
            )

        elapsed = time.monotonic() - start_time

        # Record completion custody event
        self._record_custody_event(
            event_type="pii_key_rotation_completed",
            data={
                "action": "rotate_pii_key",
                "total_records": total,
                "rotated": rotated,
                "failed": failed,
                "elapsed_seconds": round(elapsed, 2),
                "operator": getpass.getuser(),
                "timestamp": timezone.now().isoformat(),
            },
        )

        status = self.style.SUCCESS if failed == 0 else self.style.WARNING
        self.stdout.write(status(
            f"Key rotation complete: {rotated} rotated, {failed} failed "
            f"out of {total} records ({elapsed:.1f}s)."
        ))

    def _record_custody_event(self, event_type, data):
        """Create a CustodyEvent for the key rotation audit trail."""
        try:
            job = Job.objects.first()
            if job is not None:
                CustodyEvent.objects.create(
                    document_id=f"pii-key-rotation-{timezone.now().isoformat()}",
                    job=job,
                    event_type=event_type,
                    data=data,
                )
            else:
                logger.info(
                    "No Job available for custody event — logging: %s %s",
                    event_type,
                    data,
                )
        except Exception:
            logger.exception("Failed to record custody event for key rotation")
