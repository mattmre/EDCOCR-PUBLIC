"""Management command to archive and rotate old CustodyEvent records.

Exports CustodyEvent records older than the retention window to a JSONL
archive file, verifies the archive's hash-chain integrity, then deletes
the archived records from the database.

Supports:
- ``--retention-days N``: Override default from ``AUDIT_LOG_RETENTION_DAYS``
  env var (default: 2555 days = ~7 years).
- ``--archive-dir <path>``: Directory for JSONL archive files (default:
  ``{NFS_ROOT}/archive/``).
- ``--dry-run``: Preview records that would be archived without changing
  anything.
- ``--confirm``: Required safety guard for actual archival + deletion.
- LITIGATION_HOLD env var blocks all automated operations.

Emits a CustodyEvent (``audit_logs_rotated``) recording the rotation.

Usage:
    python manage.py rotate_audit_logs --dry-run
    python manage.py rotate_audit_logs --retention-days 3650 --confirm
    python manage.py rotate_audit_logs --archive-dir /backups/audit --confirm
"""

import getpass
import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone

from jobs.litigation_hold import check_litigation_hold
from jobs.models import CustodyArchiveHead, CustodyEvent
from jobs.rls import control_plane_rls_command

logger = logging.getLogger(__name__)

_DEFAULT_AUDIT_LOG_RETENTION_DAYS = 2555  # ~7 years
_MAX_ARCHIVE_RECORD_BYTES = 1 * 1024 * 1024


def _get_retention_days():
    """Read AUDIT_LOG_RETENTION_DAYS env var, returning default on parse failure."""
    raw = os.environ.get("AUDIT_LOG_RETENTION_DAYS")
    if raw is None:
        return _DEFAULT_AUDIT_LOG_RETENTION_DAYS
    try:
        return int(raw)
    except (ValueError, TypeError):
        logger.warning(
            "Invalid value for AUDIT_LOG_RETENTION_DAYS: %r -- using default %d",
            raw,
            _DEFAULT_AUDIT_LOG_RETENTION_DAYS,
        )
        return _DEFAULT_AUDIT_LOG_RETENTION_DAYS


def _get_default_archive_dir():
    """Return the default archive directory path."""
    nfs_root = getattr(settings, "NFS_ROOT", "/shared")
    return os.path.join(nfs_root, "archive")


def _serialize_event(event):
    """Serialize a CustodyEvent model instance to a JSON-safe dict.

    Preserves the hash chain fields (prev_hash, event_hash) so the
    archive can be independently verified.
    """
    job_id = str(event.job_id) if getattr(event, "job_id", None) else ""
    tenant_id = str(getattr(event, "tenant_id", "") or "")
    chain_id = str(getattr(event, "chain_id", "") or "").strip()
    if not chain_id:
        identity = (
            f"job\0{job_id}\0{event.document_id}"
            if job_id
            else f"legacy\0{tenant_id}\0{event.document_id}\0{event.event_hash}"
        )
        chain_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return {
        "id": event.id,
        "document_id": event.document_id,
        # Control-plane events are not document/job events.  Preserve a blank
        # job_id rather than serializing Python's ``None`` as a fake job.
        "job_id": job_id,
        "tenant_id": tenant_id,
        "chain_id": chain_id,
        "hash_version": int(getattr(event, "hash_version", 1) or 1),
        "event_type": event.event_type,
        "timestamp": event.timestamp.isoformat()
        if hasattr(event.timestamp, "isoformat")
        else str(event.timestamp),
        "worker_hostname": getattr(event, "worker_hostname", ""),
        "data": event.data if isinstance(event.data, dict) else {},
        "prev_hash": getattr(event, "prev_hash", ""),
        "event_hash": getattr(event, "event_hash", ""),
        "chain_finalized": getattr(event, "chain_finalized", False),
    }


def _write_archive(events, archive_path, *, chain_metadata=None):
    """Write a list of CustodyEvent instances to a JSONL archive file.

    Returns the number of records written.
    """
    archive_dir = os.path.dirname(archive_path) or "."
    os.makedirs(archive_dir, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(archive_path)}.",
        suffix=".tmp",
        dir=archive_dir,
    )
    count = 0
    seen_chains = set()
    chain_metadata = chain_metadata or {}
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            fd = -1
            for event in events:
                record = _serialize_event(event)
                chain_id = record["chain_id"]
                if chain_id not in seen_chains:
                    # Explicitly bind an archive segment to its predecessor.
                    # Blank means genesis; nonblank means the predecessor is
                    # retained or lives in a prior checksum-sealed archive.
                    metadata = chain_metadata.get(chain_id, {})
                    record["chain_anchor"] = metadata.get(
                        "chain_anchor", record.get("prev_hash") or ""
                    )
                    record["previous_archive_checksum"] = metadata.get(
                        "previous_archive_checksum", ""
                    )
                    seen_chains.add(chain_id)
                encoded = json.dumps(record, sort_keys=True, default=str)
                if len(encoded.encode("utf-8")) > _MAX_ARCHIVE_RECORD_BYTES:
                    raise ValueError(
                        "custody archive record exceeds the 1 MiB safety bound"
                    )
                f.write(encoded + "\n")
                count += 1
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, archive_path)
        # Make the rename durable where the filesystem supports directory
        # descriptors.  The archive must not appear verified and then vanish
        # on a power loss before the database rows are deleted.
        try:
            dir_fd = os.open(archive_dir, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            logger.warning("Could not fsync audit archive directory %s", archive_dir)
    except Exception:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise
    return count


def _verify_archive(archive_path, *, expected_records=None):
    """Verify the integrity of a JSONL archive file.

    Checks that:
    1. The file is valid JSONL (each line is parseable JSON).
    2. Hash-chained events have consistent prev_hash linkage where
       event_hash is non-empty.

    Returns (is_valid, message).
    """
    try:
        previous_id = None
        chain_heads = {}
        record_count = 0
        with open(archive_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                if len(line.encode("utf-8")) > _MAX_ARCHIVE_RECORD_BYTES:
                    return (
                        False,
                        f"Archive record exceeds size bound at line {line_num}",
                    )
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    return False, f"Invalid JSON at line {line_num}: {exc}"
                record_count += 1
                index = record_count

                if expected_records is not None:
                    if index > len(expected_records):
                        return (
                            False,
                            "Archive contains records outside the locked snapshot",
                        )
                    expected = expected_records[index - 1]
                    for field in (
                        "id",
                        "document_id",
                        "job_id",
                        "event_hash",
                        "prev_hash",
                        "chain_id",
                        "hash_version",
                    ):
                        if record.get(field) != expected.get(field):
                            return False, f"Archive {field} mismatch at line {index}"

                record_id = record.get("id")
                if (
                    isinstance(record_id, bool)
                    or not isinstance(record_id, int)
                    or record_id <= 0
                ):
                    return False, f"Invalid custody ID at line {index}"
                if previous_id is not None and record_id <= previous_id:
                    return False, f"Custody ID order break at line {index}"
                previous_id = record_id

                job_id = record.get("job_id")
                document_id = record.get("document_id")
                if not isinstance(job_id, str):
                    return False, f"Invalid custody job_id at line {index}"
                if not isinstance(document_id, str) or not document_id:
                    return False, f"Missing custody document_id at line {index}"
                chain_id = record.get("chain_id")
                if (
                    not isinstance(chain_id, str)
                    or len(chain_id) != 64
                    or any(char not in "0123456789abcdef" for char in chain_id)
                ):
                    return False, f"Invalid custody chain_id at line {index}"
                hash_version = record.get("hash_version", 1)
                if hash_version not in (1, 2):
                    return False, f"Invalid custody hash_version at line {index}"
                expected_prev = chain_heads.get(chain_id)
                record_prev = record.get("prev_hash")
                if expected_prev is None:
                    anchor = record.get("chain_anchor", "")
                    if not isinstance(anchor, str):
                        return False, f"Invalid custody chain anchor at line {index}"
                    if anchor and (
                        len(anchor) != 64
                        or any(char not in "0123456789abcdef" for char in anchor)
                    ):
                        return False, f"Invalid custody chain anchor at line {index}"
                    if (record_prev or "") != anchor:
                        return False, f"Custody genesis break at line {index}"
                    prior_checksum = record.get("previous_archive_checksum", "")
                    if not isinstance(prior_checksum, str) or (
                        prior_checksum
                        and (
                            len(prior_checksum) != 64
                            or any(
                                char not in "0123456789abcdef"
                                for char in prior_checksum
                            )
                        )
                    ):
                        return False, (
                            f"Invalid previous archive checksum at line {index}"
                        )
                    hash_prev = anchor or None
                else:
                    if record_prev != expected_prev:
                        return False, f"Custody prev_hash break at line {index}"
                    hash_prev = expected_prev

                event_hash = record.get("event_hash")
                if (
                    not isinstance(event_hash, str)
                    or len(event_hash) != 64
                    or any(char not in "0123456789abcdef" for char in event_hash)
                ):
                    return False, f"Invalid custody event_hash at line {index}"
                if record.get("chain_finalized") is not True:
                    return False, f"Unfinalized custody event at line {index}"
                data = record.get("data")
                if not isinstance(data, dict):
                    return False, f"Invalid custody data at line {index}"
                try:
                    timestamp = datetime.fromisoformat(record["timestamp"])
                except (KeyError, TypeError, ValueError):
                    return False, f"Invalid custody timestamp at line {index}"
                event_type = record.get("event_type")
                if not isinstance(event_type, str) or not event_type:
                    return False, f"Invalid custody event_type at line {index}"
                event_dict = {
                    "document_id": document_id,
                    "event_type": event_type,
                    "timestamp": timestamp.isoformat(timespec="milliseconds"),
                    "data": data,
                    "prev_hash": hash_prev,
                }
                if hash_version >= 2:
                    event_dict = {
                        **event_dict,
                        "chain_id": chain_id,
                        "hash_version": hash_version,
                    }
                expected_hash = hashlib.sha256(
                    json.dumps(
                        event_dict,
                        sort_keys=True,
                        default=str,
                    ).encode("utf-8")
                ).hexdigest()
                if event_hash != expected_hash:
                    return False, f"Custody event_hash mismatch at line {index}"
                chain_heads[chain_id] = event_hash

        if expected_records is not None and record_count != len(expected_records):
            return False, "Archive record count differs from the locked snapshot"
        if not record_count:
            return True, "Empty archive"
        return True, f"Archive verified: {record_count} records"

    except OSError as exc:
        return False, f"Failed to read archive: {exc}"


def _compute_archive_checksum(archive_path):
    """Compute SHA-256 checksum of the archive file."""
    sha256 = hashlib.sha256()
    with open(archive_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _archive_segment_path(archive_dir, expected_records, chain_metadata):
    """Return a content-addressed path that a failed rotation can reuse."""
    fingerprint = json.dumps(
        {"records": expected_records, "chain_metadata": chain_metadata},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    digest = hashlib.sha256(fingerprint).hexdigest()
    return os.path.join(
        archive_dir,
        (
            f"custody_events_{expected_records[0]['id']}_"
            f"{expected_records[-1]['id']}_{digest[:24]}.jsonl"
        ),
    )


def _stage_or_reuse_archive(events, archive_path, expected_records, chain_metadata):
    """Write once, or validate and reuse a prior crash-orphaned segment."""
    if os.path.exists(archive_path):
        valid, message = _verify_archive(
            archive_path,
            expected_records=expected_records,
        )
        if not valid:
            raise CommandError(
                "Existing deterministic custody archive is invalid: " + message
            )
        return len(expected_records)
    written = _write_archive(events, archive_path, chain_metadata=chain_metadata)
    valid, message = _verify_archive(
        archive_path,
        expected_records=expected_records,
    )
    if not valid:
        raise CommandError(
            f"Archive verification failed: {message}. Records NOT deleted from database."
        )
    return written


def _record_rotation_custody_event(
    record_count, archive_path, archive_checksum, retention_days
):
    """Create a CustodyEvent recording an audit log rotation action."""
    data = {
        "action": "rotate_audit_logs",
        "retention_policy": f"AUDIT_LOG_RETENTION_DAYS={retention_days}",
        "records_archived": record_count,
        # Preserve the artifact name for audit correlation without storing a
        # host/container filesystem path in the durable custody payload.
        "archive_path": os.path.basename(archive_path),
        "archive_checksum": archive_checksum,
        "reason": "retention_policy",
        "operator": getpass.getuser(),
    }

    try:
        from jobs.tasks import _record_control_plane_custody_event

        _record_control_plane_custody_event(
            tenant_id="__control_plane__",
            document_id="audit-rotation-control-plane",
            event_type="audit_logs_rotated",
            data=data,
            worker_hostname="management-command",
        )
    except Exception:
        logger.exception("Failed to record audit rotation custody event")
        return False
    return True


class Command(BaseCommand):
    help = "Archive and rotate old CustodyEvent records (respects LITIGATION_HOLD)"

    def add_arguments(self, parser):
        parser.add_argument(
            "--retention-days",
            type=int,
            default=_get_retention_days(),
            help=(
                "Archive CustodyEvent records older than this many days "
                f"(default: {_DEFAULT_AUDIT_LOG_RETENTION_DAYS}, "
                "overridable via AUDIT_LOG_RETENTION_DAYS env)"
            ),
        )
        parser.add_argument(
            "--archive-dir",
            type=str,
            default=_get_default_archive_dir(),
            help="Directory for JSONL archive files (default: {NFS_ROOT}/archive/)",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Preview what would be archived without changing anything",
        )
        parser.add_argument(
            "--confirm",
            action="store_true",
            help="Required for actual archival and deletion (safety guard)",
        )

    @control_plane_rls_command
    def handle(self, *args, **options):
        # ---- Litigation hold gate -----------------------------------------
        if check_litigation_hold(self):
            return

        retention_days = options["retention_days"]
        archive_dir = options["archive_dir"]
        dry_run = options["dry_run"]
        confirm = options["confirm"]

        if retention_days <= 0:
            raise CommandError("Audit log retention days must be positive")

        cutoff = timezone.now() - timezone.timedelta(days=retention_days)

        # Find CustodyEvent records older than the retention period
        old_events = CustodyEvent.objects.filter(timestamp__lt=cutoff)
        count = old_events.count()

        if count == 0:
            self.stdout.write(
                f"No CustodyEvent records older than {retention_days} days."
            )
            return

        # ---- Display summary ----------------------------------------------
        self.stdout.write(
            f"Found {count} CustodyEvent records older than {retention_days} days."
        )

        # ---- Dry run stops here -------------------------------------------
        if dry_run:
            self.stdout.write("DRY RUN: No records archived or deleted.")
            return

        # ---- Confirm gate -------------------------------------------------
        if not confirm:
            self.stdout.write(
                self.style.WARNING(
                    "Use --confirm to actually archive and delete these records."
                )
            )
            return

        # Stage a deterministic archive before opening the short database
        # commit. If the process or commit fails later, retry computes the same
        # path, verifies the existing bytes, and resumes instead of creating a
        # duplicate orphan archive.
        snapshot_events = list(
            CustodyEvent.objects.filter(timestamp__lt=cutoff).order_by("id")
        )
        if not snapshot_events:
            self.stdout.write(
                f"No CustodyEvent records older than {retention_days} days."
            )
            return
        expected_records = [_serialize_event(event) for event in snapshot_events]
        chain_segments = {}
        for event, record in zip(snapshot_events, expected_records, strict=True):
            chain_segments.setdefault(record["chain_id"], []).append(event)
        head_snapshot = {
            head.chain_id: head
            for head in CustodyArchiveHead.objects.filter(chain_id__in=chain_segments)
        }
        chain_metadata = {}
        for chain_id, segment in chain_segments.items():
            head = head_snapshot.get(chain_id)
            expected_anchor = head.event_hash if head is not None else ""
            first_prev = str(segment[0].prev_hash or "")
            if first_prev != expected_anchor:
                raise CommandError(
                    "Audit rotation blocked: chain segment does not continue "
                    f"the durable archive head for {chain_id}"
                )
            chain_metadata[chain_id] = {
                "chain_anchor": expected_anchor,
                "previous_archive_checksum": (
                    head.archive_checksum if head is not None else ""
                ),
            }
        archive_path = _archive_segment_path(
            archive_dir,
            expected_records,
            chain_metadata,
        )
        written = _stage_or_reuse_archive(
            snapshot_events,
            archive_path,
            expected_records,
            chain_metadata,
        )
        archive_checksum = _compute_archive_checksum(archive_path)
        candidate_ids = [event.id for event in snapshot_events]

        with transaction.atomic():
            locked_events = list(
                CustodyEvent.objects.filter(id__in=candidate_ids)
                .order_by("id")
                .select_for_update()
            )
            if [_serialize_event(event) for event in locked_events] != expected_records:
                raise CommandError(
                    "Audit rotation blocked: database snapshot changed after staging"
                )
            locked_heads = {
                head.chain_id: head
                for head in CustodyArchiveHead.objects.select_for_update().filter(
                    chain_id__in=chain_segments
                )
            }
            for chain_id, metadata in chain_metadata.items():
                head = locked_heads.get(chain_id)
                current = {
                    "chain_anchor": head.event_hash if head is not None else "",
                    "previous_archive_checksum": (
                        head.archive_checksum if head is not None else ""
                    ),
                }
                if current != metadata:
                    raise CommandError(
                        "Audit rotation blocked: archive head changed after staging"
                    )

            for chain_id, segment in chain_segments.items():
                previous_head = locked_heads.get(chain_id)
                CustodyArchiveHead.objects.update_or_create(
                    chain_id=chain_id,
                    defaults={
                        "event_hash": segment[-1].event_hash,
                        "last_event_id": segment[-1].id,
                        "archive_path": archive_path,
                        "archive_checksum": archive_checksum,
                        "legacy_unverified": bool(
                            previous_head and previous_head.legacy_unverified
                        ),
                    },
                )

            if not _record_rotation_custody_event(
                record_count=len(locked_events),
                archive_path=archive_path,
                archive_checksum=archive_checksum,
                retention_days=retention_days,
            ):
                raise CommandError(
                    "Audit rotation blocked: custody event could not be recorded"
                )
            deleted_count, _ = CustodyEvent.objects.filter(
                id__in=candidate_ids
            ).delete()
            if deleted_count != len(candidate_ids):
                raise CommandError(
                    "Audit rotation blocked: database deletion did not match "
                    "the locked archive snapshot"
                )

        self.stdout.write(
            self.style.SUCCESS(
                f"Archived {written} records to {archive_path} "
                f"(checksum: {archive_checksum[:16]}...). "
                f"Deleted {deleted_count} records from database."
            )
        )
