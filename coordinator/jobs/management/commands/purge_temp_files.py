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
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from jobs.litigation_hold import check_litigation_hold
from jobs.models import Job
from jobs.rls import control_plane_rls_command
from ocr_distributed.tenant_purge_fs import remove_path_no_follow

logger = logging.getLogger(__name__)


def _record_purge_custody_event(
    directory, size_bytes, event_type="temp_files_purged", **extra
):
    """Create a CustodyEvent recording a temp-file purge action.

    This is a control-plane event and therefore has no document/job anchor.
    Associating it with an arbitrary first job would misstate provenance.
    """
    try:
        from jobs.tasks import _record_control_plane_custody_event

        _record_control_plane_custody_event(
            tenant_id="__control_plane__",
            document_id="temp-purge-control-plane",
            event_type=event_type,
            data={
                "action": "temp_purge",
                "directory": directory,
                "size_bytes": size_bytes,
                "reason": "orphaned_directory",
                "operator": getpass.getuser(),
                **extra,
            },
            worker_hostname="management-command",
        )
    except Exception:
        logger.exception("Failed to record purge custody event")
        return False
    return True


class Command(BaseCommand):
    help = "Remove orphaned NFS job directories without matching Job records"

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Preview deletions without actually deleting",
        )

    @control_plane_rls_command
    def handle(self, *args, **options):
        # ---- Litigation hold gate -----------------------------------------
        if check_litigation_hold(self):
            return

        dry_run = options["dry_run"]
        nfs_root = Path(settings.NFS_ROOT).resolve(strict=False)
        raw_jobs_dir = Path(settings.NFS_ROOT) / "jobs"
        jobs_dir = raw_jobs_dir.resolve(strict=False)
        try:
            jobs_dir.relative_to(nfs_root)
        except ValueError as exc:
            raise CommandError(
                "Cleanup blocked: NFS jobs directory escapes NFS_ROOT"
            ) from exc

        if not jobs_dir.is_dir():
            self.stdout.write(f"NFS jobs directory does not exist: {jobs_dir}")
            return

        # List all directories in the NFS jobs path
        nfs_dirs = []
        blocked_paths = []
        try:
            with os.scandir(jobs_dir) as entries:
                for entry in entries:
                    full_path = os.path.join(str(jobs_dir), entry.name)
                    if entry.is_symlink():
                        blocked_paths.append(
                            f"{entry.name}: symlinked NFS job directory"
                        )
                    elif entry.is_dir(follow_symlinks=False):
                        nfs_dirs.append((entry.name, full_path))
        except OSError as exc:
            raise CommandError(
                f"Cleanup blocked: cannot inventory NFS jobs directory: {exc}"
            ) from exc

        if not nfs_dirs and not blocked_paths:
            self.stdout.write(f"No directories found in {jobs_dir}.")
            return

        # Check which directories have matching Job records
        existing_job_ids = set(
            str(jid) for jid in Job.objects.values_list("job_id", flat=True)
        )

        orphans = [
            (name, path) for name, path in nfs_dirs if name not in existing_job_ids
        ]

        if blocked_paths and not dry_run:
            raise CommandError(
                "Cleanup blocked: refusing symlinked NFS entries; "
                + "; ".join(blocked_paths[:5])
            )

        if not orphans and not blocked_paths:
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
            for blocked in blocked_paths[:10]:
                self.stdout.write(f"  REFUSED: {blocked}")
            return

        removed = 0
        cleanup_failures = []
        for _, path in orphans:
            size_bytes = orphan_sizes.get(path, 0)
            if not _record_purge_custody_event(
                path,
                size_bytes,
                "temp_files_deletion_started",
            ):
                cleanup_failures.append(f"{path}: deletion-start custody failed")
                continue
            try:
                removed_safely = remove_path_no_follow(path)
            except (OSError, RuntimeError) as exc:
                cleanup_failures.append(f"{path}: {exc}")
                _record_purge_custody_event(
                    path,
                    size_bytes,
                    "temp_files_deletion_failed",
                    failure=str(exc),
                )
                continue
            if not removed_safely or os.path.lexists(path):
                cleanup_failures.append(f"{path}: directory remains")
                _record_purge_custody_event(
                    path,
                    size_bytes,
                    "temp_files_deletion_failed",
                    failure="directory remains",
                )
                continue
            if not _record_purge_custody_event(
                path,
                size_bytes,
                "temp_files_purged",
            ):
                cleanup_failures.append(f"{path}: custody event could not be recorded")
                continue
            removed += 1

        if cleanup_failures:
            raise CommandError(
                "Cleanup blocked: orphan removal failed; "
                + "; ".join(cleanup_failures[:5])
            )

        self.stdout.write(
            self.style.SUCCESS(
                f"Removed {removed} orphaned directories ({size_mb:.1f} MB recovered)."
            )
        )
