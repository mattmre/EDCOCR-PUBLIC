"""Management command to rotate the PII encryption key (R2-RC-21).

Re-encrypts **all** Fernet-backed fields:
- ``PiiEntity.entity_value``
- ``ExtractedEntity.entity_text``
- ``ExtractedFormValue.field_value``
- ``PresignedTaskCapability.payload``

Reads **raw** DB ciphertext (bypasses ``EncryptedTextField.from_db_value``) so
rotation works even when ``PII_ENCRYPTION_KEY`` already points at the new key.
Writes through a parameterized raw ``UPDATE`` so Django cannot invoke
``EncryptedTextField.get_prep_value`` and encrypt the rotated ciphertext again.

After a successful run, calls ``reset_encryptor()`` so the process singleton
reloads from env.  Any per-record failure yields a **non-zero exit code**.

Usage:
    PII_ENCRYPTION_KEY=<new> python manage.py rotate_pii_key \
      --old-key <base64_key> --new-key <base64_key> --confirm-writers-drained
    python manage.py rotate_pii_key --old-key <key> --new-key <key> --dry-run
    python manage.py rotate_pii_key --old-key <key> --new-key <key> --batch-size 500
"""

from __future__ import annotations

import getpass
import logging
import os
import time
from typing import List, Sequence, Tuple

from django.core.management.base import BaseCommand, CommandError
from django.db import connection, models, transaction
from django.utils import timezone

from jobs.extraction_models import ExtractedEntity, ExtractedFormValue
from jobs.models import PiiEntity, PresignedTaskCapability
from jobs.pii_encryption import PiiEncryptor, reset_encryptor
from jobs.rls import control_plane_rls_command

logger = logging.getLogger(__name__)

_DEFAULT_BATCH_SIZE = 1000

# (model, field_name) — every EncryptedTextField in the coordinator schema.
_ENCRYPTED_TARGETS: Sequence[Tuple[type[models.Model], str]] = (
    (PiiEntity, "entity_value"),
    (ExtractedEntity, "entity_text"),
    (ExtractedFormValue, "field_value"),
    (PresignedTaskCapability, "payload"),
)


class Command(BaseCommand):
    help = (
        "Re-encrypt all PII/PHI Fernet fields with a new encryption key "
        "(PiiEntity, ExtractedEntity, ExtractedFormValue). R2-RC-21 fail-closed."
    )

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
        parser.add_argument(
            "--confirm-writers-drained",
            action="store_true",
            help="Confirm every PII/capability writer is stopped for key cutover",
        )

    @control_plane_rls_command
    def handle(self, *args, **options):
        old_key = options["old_key"]
        new_key = options["new_key"]
        batch_size = options["batch_size"]
        dry_run = options["dry_run"]
        writers_drained = options["confirm_writers_drained"]

        if old_key == new_key:
            self.stderr.write(
                self.style.WARNING("Old and new keys are identical. Nothing to do.")
            )
            return
        if not dry_run and not writers_drained:
            raise CommandError(
                "Key rotation requires --confirm-writers-drained; concurrent "
                "old-key writes would make the cutover inconsistent"
            )
        if not dry_run and os.environ.get("PII_ENCRYPTION_KEY", "").strip() != new_key:
            raise CommandError(
                "PII_ENCRYPTION_KEY must already equal --new-key in the drained "
                "rotation process"
            )

        try:
            old_encryptor = PiiEncryptor(old_key)
            if not old_encryptor.enabled:
                raise CommandError(
                    "Old key is invalid or cryptography is not installed."
                )
        except CommandError:
            raise
        except Exception as exc:
            raise CommandError(f"Invalid old key: {exc}") from exc

        try:
            new_encryptor = PiiEncryptor(new_key)
            if not new_encryptor.enabled:
                raise CommandError("New key is invalid.")
        except CommandError:
            raise
        except Exception as exc:
            raise CommandError(f"Invalid new key: {exc}") from exc

        counts = {
            f"{model._meta.label}.{field}": model.objects.count()
            for model, field in _ENCRYPTED_TARGETS
        }
        total = sum(counts.values())
        if total == 0:
            self.stdout.write("No encrypted PII/PHI records to rotate.")
            return

        self.stdout.write(
            "Targets: "
            + ", ".join(f"{label}={n}" for label, n in counts.items())
            + f" (total={total})"
        )

        if dry_run:
            self.stdout.write(
                f"DRY RUN: Would re-encrypt {total} records in batches of {batch_size}."
            )
            return

        if not self._record_custody_event(
            event_type="pii_key_rotation_started",
            data={
                "action": "rotate_pii_key",
                "total_records": total,
                "counts": counts,
                "batch_size": batch_size,
                "operator": getpass.getuser(),
                "timestamp": timezone.now().isoformat(),
                "r2_rc_21": True,
            },
        ):
            raise CommandError(
                "Key rotation blocked: start custody event could not be recorded"
            )

        rotated = 0
        failed = 0
        skipped = 0
        start_time = time.monotonic()
        elapsed = 0.0

        try:
            # All encrypted columns move together or none do. Operators must
            # still drain writers before invoking the command; this database
            # transaction prevents a partial mixed-key state on any failure.
            with transaction.atomic():
                if connection.vendor == "postgresql":
                    tables = sorted(
                        {
                            connection.ops.quote_name(model._meta.db_table)
                            for model, _ in _ENCRYPTED_TARGETS
                        }
                    )
                    with connection.cursor() as cursor:
                        cursor.execute(
                            "LOCK TABLE "
                            + ", ".join(tables)
                            + " IN ACCESS EXCLUSIVE MODE"
                        )
                for model, field_name in _ENCRYPTED_TARGETS:
                    r, f, s = self._rotate_model(
                        model=model,
                        field_name=field_name,
                        old_encryptor=old_encryptor,
                        new_encryptor=new_encryptor,
                        old_key=old_key,
                        new_key=new_key,
                        batch_size=batch_size,
                    )
                    rotated += r
                    failed += f
                    skipped += s
                    self.stdout.write(
                        f"  {model._meta.label}.{field_name}: "
                        f"{r} rotated, {f} failed, {s} skipped empty"
                    )
                if failed:
                    raise CommandError(
                        f"rotate_pii_key failed for {failed} record(s); "
                        "all key updates were rolled back"
                    )
                elapsed = time.monotonic() - start_time
                if not self._record_custody_event(
                    event_type="pii_key_rotation_completed",
                    data={
                        "action": "rotate_pii_key",
                        "total_records": total,
                        "rotated": rotated,
                        "failed": failed,
                        "skipped_empty": skipped,
                        "elapsed_seconds": round(elapsed, 2),
                        "operator": getpass.getuser(),
                        "timestamp": timezone.now().isoformat(),
                        "r2_rc_21": True,
                    },
                ):
                    raise CommandError(
                        "Key rotation incomplete: completion custody event "
                        "could not be recorded; all key updates were rolled back"
                    )
        except Exception as exc:
            self._record_custody_event(
                event_type="pii_key_rotation_failed",
                data={
                    "action": "rotate_pii_key",
                    "failed": failed,
                    "error": f"{type(exc).__name__}: {exc}",
                    "operator": getpass.getuser(),
                    "timestamp": timezone.now().isoformat(),
                },
            )
            if isinstance(exc, CommandError):
                raise
            raise CommandError(f"PII key rotation rolled back: {exc}") from exc

        # Force process-local encryptor to re-read env after rotation.
        reset_encryptor()

        self.stdout.write(
            self.style.SUCCESS(
                f"Key rotation complete: {rotated} rotated, {failed} failed, "
                f"{skipped} empty skipped of {total} ({elapsed:.1f}s)."
            )
        )

    def _rotate_model(
        self,
        *,
        model: type[models.Model],
        field_name: str,
        old_encryptor: PiiEncryptor,
        new_encryptor: PiiEncryptor,
        old_key: str,
        new_key: str,
        batch_size: int,
    ) -> Tuple[int, int, int]:
        """Rotate one model's encrypted column via raw SQL reads.

        Returns (rotated, failed, skipped_empty).
        """
        table = connection.ops.quote_name(model._meta.db_table)
        pk_col = connection.ops.quote_name(model._meta.pk.column)
        field = model._meta.get_field(field_name)
        val_col = connection.ops.quote_name(field.column)

        rotated = 0
        failed = 0
        skipped = 0
        last_pk = None

        while True:
            rows = self._fetch_raw_batch(
                table=table,
                pk_col=pk_col,
                val_col=val_col,
                last_pk=last_pk,
                batch_size=batch_size,
            )
            if not rows:
                break

            for pk, raw_value in rows:
                last_pk = pk
                if raw_value is None or raw_value == "":
                    skipped += 1
                    continue
                if not isinstance(raw_value, str):
                    raw_value = str(raw_value)

                try:
                    if PiiEncryptor.is_rotation_ciphertext_candidate(raw_value):
                        new_value = old_encryptor.rotate_key(
                            old_key, new_key, raw_value
                        )
                    else:
                        # Legacy plaintext row: encrypt under new key only.
                        new_value = new_encryptor.encrypt(raw_value)

                    self._write_raw_value(
                        table=table,
                        pk_col=pk_col,
                        val_col=val_col,
                        pk=pk,
                        value=new_value,
                    )
                    rotated += 1
                except Exception:
                    failed += 1
                    logger.exception(
                        "Failed to rotate key for %s pk=%s field=%s",
                        model._meta.label,
                        pk,
                        field_name,
                    )

            self.stdout.write(
                f"  Progress {model._meta.label}: last_pk={last_pk} "
                f"({rotated} rotated, {failed} failed, {skipped} empty)"
            )

        return rotated, failed, skipped

    def _write_raw_value(
        self,
        *,
        table: str,
        pk_col: str,
        val_col: str,
        pk: object,
        value: str,
    ) -> None:
        """Persist already-encrypted ciphertext without Django field preparation."""
        sql = f"UPDATE {table} SET {val_col} = %s WHERE {pk_col} = %s"
        with connection.cursor() as cursor:
            cursor.execute(sql, [value, pk])
            if cursor.rowcount != 1:
                raise RuntimeError(
                    f"Key rotation update matched {cursor.rowcount} rows for pk={pk}"
                )

    def _fetch_raw_batch(
        self,
        *,
        table: str,
        pk_col: str,
        val_col: str,
        last_pk: object | None,
        batch_size: int,
    ) -> List[Tuple[int, object]]:
        """Return raw (pk, ciphertext) without EncryptedTextField decrypt."""
        where = f"WHERE {pk_col} > %s " if last_pk is not None else ""
        sql = f"SELECT {pk_col}, {val_col} FROM {table} {where}ORDER BY {pk_col} ASC LIMIT %s"
        params = [last_pk, batch_size] if last_pk is not None else [batch_size]
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            return list(cursor.fetchall())

    def _record_custody_event(self, event_type, data):
        """Create a CustodyEvent for the key rotation audit trail."""
        try:
            from jobs.tasks import _record_control_plane_custody_event

            _record_control_plane_custody_event(
                tenant_id="__control_plane__",
                document_id="pii-key-rotation-control-plane",
                event_type=event_type,
                data=data,
                worker_hostname="management-command",
            )
        except Exception:
            logger.exception("Failed to record custody event for key rotation")
            return False
        return True
