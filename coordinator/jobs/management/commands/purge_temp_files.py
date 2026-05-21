"""Management command to purge orphaned NFS temp directories.

Scans the NFS jobs directory for directories that don't correspond to
any existing Job record, and removes them.

Usage:
    python manage.py purge_temp_files              # Remove orphans
    python manage.py purge_temp_files --dry-run    # Preview only
"""

import getpass
import logging
import os
import shutil

from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils import timezone

from jobs.litigation_hold import check_litigation_hold
from jobs.models import CustodyEvent, Job

logger = logging.getLogger(__name__)


def _record_purge_custody_event(directory, size_bytes):
    """Create a CustodyEvent recording a temp-file purge action.

    Anchors to the first available Job, or logs if none exist (same
    pattern as cleanup_old_jobs.py).
    """
    try:
        first_job = Job.objects.first()
        if first_job is not None:
            CustodyEvent.objects.create(
                document_id=f"temp-purge-{timezone.now().isoformat()}",
                job=first_job,
                event_type="temp_files_purged",
                data={
                    "action": "temp_purge",
                    "directory": directory,
                    "size_bytes": size_bytes,
                    "reason": "orphaned_directory",
                    "operator": getpass.getuser(),
                },
            )
        else:
            logger.info(
                "No Job available to anchor custody event -- logging instead: "
                "temp_files_purged directory=%s size_bytes=%d",
                directory,
                size_bytes,
            )
    except Exception:
        # Custody event recording must never prevent the purge itself.
        logger.exception("Failed to record purge custody event")


class Command(BaseCommand):
    help = "Remove orphaned NFS job directories without matching Job records"

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Preview deletions without actually deleting",
        )

    def handle(self, *args, **options):
        # ---- Litigation hold gate -----------------------------------------
        if check_litigation_hold(self):
            return

        dry_run = options["dry_run"]
        jobs_dir = os.path.join(settings.NFS_ROOT, "jobs")

        if not os.path.isdir(jobs_dir):
            self.stdout.write(f"NFS jobs directory does not exist: {jobs_dir}")
            return

        # List all directories in the NFS jobs path
        nfs_dirs = []
        for entry in os.listdir(jobs_dir):
            full_path = os.path.join(jobs_dir, entry)
            if os.path.isdir(full_path):
                nfs_dirs.append((entry, full_path))

        if not nfs_dirs:
            self.stdout.write(f"No directories found in {jobs_dir}.")
            return

        # Check which directories have matching Job records
        existing_job_ids = set(
            str(jid) for jid in Job.objects.values_list("job_id", flat=True)
        )

        orphans = [
            (name, path) for name, path in nfs_dirs
            if name not in existing_job_ids
        ]

        if not orphans:
            self.stdout.write(
                f"No orphaned directories found ({len(nfs_dirs)} dirs, all have matching jobs)."
            )
            return

        # Calculate sizes per orphan directory
        orphan_sizes = {}
        total_size = 0
        for name, path in orphans:
            dir_size = 0
            for dirpath, _, filenames in os.walk(path):
                for f in filenames:
                    try:
                        dir_size += os.path.getsize(os.path.join(dirpath, f))
                    except OSError:
                        pass
            orphan_sizes[path] = dir_size
            total_size += dir_size

        size_mb = total_size / (1024 * 1024)

        if dry_run:
            self.stdout.write(
                f"DRY RUN: Would remove {len(orphans)} orphaned directories ({size_mb:.1f} MB)."
            )
            for name, path in orphans[:10]:
                self.stdout.write(f"  {path}")
            if len(orphans) > 10:
                self.stdout.write(f"  ... and {len(orphans) - 10} more")
            return

        removed = 0
        for _, path in orphans:
            shutil.rmtree(path, ignore_errors=True)
            _record_purge_custody_event(path, orphan_sizes.get(path, 0))
            removed += 1

        self.stdout.write(self.style.SUCCESS(
            f"Removed {removed} orphaned directories ({size_mb:.1f} MB recovered)."
        ))
