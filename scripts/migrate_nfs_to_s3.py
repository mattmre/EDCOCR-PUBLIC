#!/usr/bin/env python3
"""Migrate job artifacts from NFS storage to S3.

Walks the NFS jobs directory, uploads all artifacts per job to S3,
verifies checksums, then optionally marks the job as using S3 storage
and removes NFS files.

Usage:
    # Recommended: set credentials via environment variables
    export S3_ACCESS_KEY=KEY
    export S3_SECRET_KEY=SECRET

    # Dry run (default — no changes):
    python scripts/migrate_nfs_to_s3.py --nfs-root /shared --s3-endpoint http://minio:9000 \\
        --s3-bucket ocr-jobs

    # CLI credential fallback (NOT recommended — visible in process list):
    python scripts/migrate_nfs_to_s3.py --nfs-root /shared --s3-endpoint http://minio:9000 \\
        --s3-bucket ocr-jobs --s3-access-key KEY --s3-secret-key SECRET

    # Live migration with NFS cleanup:
    python scripts/migrate_nfs_to_s3.py --nfs-root /shared --s3-endpoint http://minio:9000 \\
        --s3-bucket ocr-jobs --execute --delete-nfs

    # Resume interrupted migration (skips already-uploaded files):
    python scripts/migrate_nfs_to_s3.py ... --execute --resume

    # Validate mode (enumerate files, compute checksums, check S3 connectivity):
    python scripts/migrate_nfs_to_s3.py --nfs-root /shared --s3-endpoint http://minio:9000 \\
        --s3-bucket ocr-jobs --validate --output-report report.json

    # Validate mode with sampling (quick sanity check on large datasets):
    python scripts/migrate_nfs_to_s3.py --nfs-root /shared --s3-endpoint http://minio:9000 \\
        --s3-bucket ocr-jobs --validate --sample 5
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Allow importing coordinator modules when run from project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_COORDINATOR_DIR = os.path.join(_PROJECT_ROOT, "coordinator")
if _COORDINATOR_DIR not in sys.path:
    sys.path.insert(0, _COORDINATOR_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from jobs.storage import NFSBackend, S3Backend  # noqa: E402

# ---------------------------------------------------------------------------
# Credential manager integration (vault / KMS / env fallback chain)
# ---------------------------------------------------------------------------
try:
    from credential_manager import get_credential as _get_credential  # noqa: E402
except ImportError:

    def _get_credential(key, default=None):  # type: ignore[misc]
        return os.environ.get(key, default)

# ---------------------------------------------------------------------------
# Optional psycopg2 for database backend updates (--update-db)
# ---------------------------------------------------------------------------
try:
    import psycopg2  # noqa: E402
except ImportError:
    psycopg2 = None  # type: ignore[assignment]


logger = logging.getLogger("migrate_nfs_to_s3")

# ---------------------------------------------------------------------------
# Transfer rate assumption for time estimation (bytes per second)
# ---------------------------------------------------------------------------
DEFAULT_TRANSFER_RATE_BPS = 50 * 1024 * 1024  # 50 MB/s


# ---------------------------------------------------------------------------
# MigrationReport dataclass
# ---------------------------------------------------------------------------


@dataclass
class FileManifestEntry:
    """Single file in the migration manifest."""

    nfs_path: str
    s3_key: str
    size_bytes: int
    sha256: str


@dataclass
class MigrationReport:
    """Report produced by the --validate dry-run validator."""

    total_files: int = 0
    total_bytes: int = 0
    estimated_time_seconds: float = 0.0
    file_manifest: list[FileManifestEntry] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    connectivity_status: str = "unknown"
    nfs_root: str = ""
    s3_endpoint: str = ""
    s3_bucket: str = ""
    jobs_found: int = 0
    scan_elapsed_seconds: float = 0.0

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict."""
        return {
            "total_files": self.total_files,
            "total_bytes": self.total_bytes,
            "total_bytes_human": _human_bytes(self.total_bytes),
            "estimated_time_seconds": round(self.estimated_time_seconds, 1),
            "estimated_time_human": _human_time(self.estimated_time_seconds),
            "jobs_found": self.jobs_found,
            "nfs_root": self.nfs_root,
            "s3_endpoint": self.s3_endpoint,
            "s3_bucket": self.s3_bucket,
            "connectivity_status": self.connectivity_status,
            "warnings": self.warnings,
            "scan_elapsed_seconds": round(self.scan_elapsed_seconds, 1),
            "file_manifest": [
                {
                    "nfs_path": e.nfs_path,
                    "s3_key": e.s3_key,
                    "size_bytes": e.size_bytes,
                    "sha256": e.sha256,
                }
                for e in self.file_manifest
            ],
        }

    def summary_text(self) -> str:
        """Human-readable summary string."""
        lines = [
            "=" * 60,
            "Migration Validation Report",
            "=" * 60,
            f"  NFS root:          {self.nfs_root}",
            f"  S3 endpoint:       {self.s3_endpoint}",
            f"  S3 bucket:         {self.s3_bucket}",
            f"  S3 connectivity:   {self.connectivity_status}",
            f"  Jobs discovered:   {self.jobs_found}",
            f"  Total files:       {self.total_files}",
            f"  Total size:        {_human_bytes(self.total_bytes)}",
            f"  Est. transfer:     {_human_time(self.estimated_time_seconds)}",
            f"  Scan elapsed:      {round(self.scan_elapsed_seconds, 1)}s",
        ]
        if self.warnings:
            lines.append(f"  Warnings:          {len(self.warnings)}")
            for w in self.warnings:
                lines.append(f"    - {w}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _human_bytes(n: int) -> str:
    """Convert byte count to human-readable string."""
    if n < 1024:
        return f"{n} B"
    for unit in ("KB", "MB", "GB", "TB"):
        n /= 1024.0
        if n < 1024:
            return f"{n:.1f} {unit}"
    return f"{n:.1f} PB"


def _human_time(seconds: float) -> str:
    """Convert seconds to human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {int(s)}s"
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    return f"{int(h)}h {int(m)}m {int(s)}s"


def sha256_file(path: str) -> str:
    """Return hex SHA-256 digest of a local file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_jobs(nfs_root: str) -> list[str]:
    """Return sorted list of job IDs found under {nfs_root}/jobs/."""
    jobs_dir = os.path.join(nfs_root, "jobs")
    if not os.path.isdir(jobs_dir):
        return []
    return sorted(
        d for d in os.listdir(jobs_dir)
        if os.path.isdir(os.path.join(jobs_dir, d))
    )


# ---------------------------------------------------------------------------
# State-file helpers (C-18: progress persistence)
# ---------------------------------------------------------------------------


def _load_state_file(state_file: str) -> dict | None:
    """Load migration state from a JSON file, or return ``None``."""
    if state_file and os.path.isfile(state_file):
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load state file %s: %s", state_file, exc)
    return None


def _save_state_file(state_file: str, state_data: dict) -> None:
    """Atomically write state data to a JSON file (write tmp then rename)."""
    tmp_path = state_file + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state_data, f, indent=2)
    os.replace(tmp_path, state_file)


# ---------------------------------------------------------------------------
# Database update helper (C-19: --update-db)
# ---------------------------------------------------------------------------


def _update_job_db(job_id: str, db_url: str | None = None) -> bool:
    """Update coordinator DB to set ``storage_backend='s3'`` for *job_id*.

    Returns ``True`` on success, ``False`` on failure (with warning logged).
    """
    if psycopg2 is None:
        logger.error(
            "psycopg2 is required for --update-db. "
            "Install with: pip install psycopg2-binary"
        )
        return False

    conn_url = db_url or os.environ.get("DATABASE_URL")
    if not conn_url:
        host = os.environ.get("POSTGRES_HOST", "localhost")
        port = os.environ.get("POSTGRES_PORT", "5432")
        user = os.environ.get("POSTGRES_USER", "postgres")
        password = os.environ.get("POSTGRES_PASSWORD", "")
        dbname = os.environ.get("POSTGRES_DB", "coordinator")
        conn_url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

    try:
        conn = psycopg2.connect(conn_url)
        try:
            cur = conn.cursor()
            cur.execute(
                "UPDATE jobs SET storage_backend = %s WHERE job_id = %s",
                ("s3", job_id),
            )
            conn.commit()
            cur.close()
        finally:
            conn.close()
        logger.info("DB updated: storage_backend='s3' for job %s", job_id)
        return True
    except Exception as exc:
        logger.warning("DB update failed for job %s: %s", job_id, exc)
        return False


def migrate_job(
    job_id: str,
    nfs: NFSBackend,
    s3: S3Backend,
    *,
    execute: bool = False,
    resume: bool = False,
    delete_nfs: bool = False,
    parallel: int = 1,
    state_file: str | None = None,
    _state_data: dict | None = None,
) -> dict:
    """Migrate a single job from NFS to S3.

    Returns a dict with migration statistics for this job.
    When *execute* is False (dry-run), the counters use ``_to_upload`` /
    ``_to_verify`` labels so callers can distinguish simulated counts from
    actual transfers.
    """
    prefix = f"jobs/{job_id}"
    nfs_keys = nfs.list_objects(prefix)

    # Use different counter labels for dry-run vs live mode so output is
    # never misleading about what actually happened.
    if execute:
        upload_key = "files_uploaded"
        verify_key = "files_verified"
        bytes_key = "bytes_uploaded"
    else:
        upload_key = "files_to_upload"
        verify_key = "files_to_verify"
        bytes_key = "bytes_to_upload"

    stats: dict = {
        "job_id": job_id,
        "files_found": len(nfs_keys),
        upload_key: 0,
        "files_skipped": 0,
        verify_key: 0,
        "files_deleted": 0,
        bytes_key: 0,
        "errors": [],
        "dry_run": not execute,
    }

    if not nfs_keys:
        return stats

    # C-18: State-file management – load verified set for resume
    _sd: dict | None = _state_data
    if state_file and _sd is None:
        _sd = _load_state_file(state_file) or {
            "nfs_root": "",
            "s3_bucket": "",
            "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "jobs": {},
        }

    state_verified: set[str] = set()
    if _sd and resume:
        job_state = _sd.get("jobs", {}).get(job_id, {})
        state_verified = set(job_state.get("verified_files", []))

    lock = threading.Lock()

    # -- Per-file work (safe to call from threads) -------------------------
    def _process_file(key: str) -> None:
        local_path = nfs.to_absolute_path(key)
        file_size = os.path.getsize(local_path)

        try:
            # C-18: Skip if already verified in state file
            if resume and key in state_verified:
                with lock:
                    stats["files_skipped"] += 1
                return

            # Skip if already in S3 (resume mode)
            if resume and s3.exists(key):
                with lock:
                    stats["files_skipped"] += 1
                return

            if execute:
                # Upload to S3
                s3.upload_file(local_path, key)

                # Verify: download and compare SHA-256
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".verify") as tmp:
                    tmp_path = tmp.name
                try:
                    s3.download_file(key, tmp_path)
                    nfs_hash = sha256_file(local_path)
                    s3_hash = sha256_file(tmp_path)
                    verified = nfs_hash == s3_hash
                finally:
                    os.unlink(tmp_path)

                with lock:
                    stats[upload_key] += 1
                    stats[bytes_key] += file_size
                    if verified:
                        stats[verify_key] += 1
                    else:
                        stats["errors"].append(
                            f"Checksum mismatch for {key}: NFS={nfs_hash} S3={s3_hash}"
                        )

                # C-18: Persist verified file to state
                if state_file and _sd is not None and verified:
                    with lock:
                        jobs_s = _sd.setdefault("jobs", {})
                        js = jobs_s.setdefault(job_id, {
                            "status": "in_progress",
                            "verified_files": [],
                            "deleted_files": [],
                        })
                        js["verified_files"].append(key)
                        js["status"] = "in_progress"
                        _save_state_file(state_file, _sd)
            else:
                # Dry run — just count what *would* happen
                with lock:
                    stats[upload_key] += 1
                    stats[bytes_key] += file_size
                    stats[verify_key] += 1
        except Exception as exc:
            with lock:
                stats["errors"].append(f"Error uploading {key}: {exc}")

    # C-17: Parallel or sequential file processing
    if parallel > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = [pool.submit(_process_file, key) for key in nfs_keys]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    with lock:
                        stats["errors"].append(f"Thread error: {exc}")
    else:
        for key in nfs_keys:
            _process_file(key)

    # Delete NFS files only if all files were verified
    if delete_nfs and execute and stats.get("files_verified", 0) == stats["files_found"] - stats["files_skipped"]:
        job_path = nfs.to_absolute_path(prefix)
        try:
            import shutil
            shutil.rmtree(job_path)
            stats["files_deleted"] = stats["files_found"]
            # C-18: Record deleted files in state
            if state_file and _sd is not None:
                js = _sd.get("jobs", {}).get(job_id)
                if js:
                    js["deleted_files"] = list(nfs_keys)
                    js["status"] = "completed"
                    _save_state_file(state_file, _sd)
        except Exception as exc:
            stats["errors"].append(f"Error deleting NFS job dir: {exc}")

    # C-18: Update final job status in state
    if state_file and _sd is not None and execute:
        jobs_s = _sd.setdefault("jobs", {})
        js = jobs_s.setdefault(job_id, {
            "status": "in_progress",
            "verified_files": [],
            "deleted_files": [],
        })
        if stats["errors"]:
            js["status"] = "failed"
        elif js.get("status") != "completed":
            expected = stats["files_found"] - stats["files_skipped"]
            if stats.get("files_verified", 0) >= expected > 0:
                js["status"] = "completed"
        _save_state_file(state_file, _sd)

    return stats


def run_migration(
    nfs_root: str,
    s3_endpoint: str,
    s3_bucket: str,
    s3_access_key: str,
    s3_secret_key: str,
    s3_region: str = "",
    *,
    execute: bool = False,
    resume: bool = False,
    delete_nfs: bool = False,
    job_ids: list[str] | None = None,
    parallel: int = 1,
    state_file: str | None = None,
    update_db: bool = False,
    db_url: str | None = None,
) -> dict:
    """Run the full NFS-to-S3 migration.

    Returns a summary dict with per-job stats and totals.
    """
    nfs = NFSBackend(root=nfs_root)
    s3 = S3Backend(
        endpoint=s3_endpoint,
        bucket=s3_bucket,
        access_key=s3_access_key,
        secret_key=s3_secret_key,
        region=s3_region,
    )

    if job_ids:
        all_job_ids = job_ids
    else:
        all_job_ids = discover_jobs(nfs_root)

    mode = "LIVE" if execute else "DRY RUN"
    logger.info("Migration mode: %s | Jobs found: %d", mode, len(all_job_ids))

    # Use different counter labels for dry-run vs live mode
    if execute:
        upload_key = "files_uploaded"
        verify_key = "files_verified"
        bytes_key = "bytes_uploaded"
    else:
        upload_key = "files_to_upload"
        verify_key = "files_to_verify"
        bytes_key = "bytes_to_upload"

    results = []
    totals: dict = {
        "mode": mode,
        "jobs_total": len(all_job_ids),
        "jobs_migrated": 0,
        "jobs_with_errors": 0,
        "files_total": 0,
        upload_key: 0,
        "files_skipped": 0,
        verify_key: 0,
        "files_deleted": 0,
        bytes_key: 0,
    }

    # C-18: Initialise state-file data
    state_data: dict | None = None
    if state_file:
        state_data = _load_state_file(state_file)
        if state_data is None:
            state_data = {
                "nfs_root": nfs_root,
                "s3_bucket": s3_bucket,
                "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "jobs": {},
            }
            _save_state_file(state_file, state_data)

    start = time.monotonic()

    for i, job_id in enumerate(all_job_ids, 1):
        logger.info("[%d/%d] Migrating job %s...", i, totals["jobs_total"], job_id)
        stats = migrate_job(
            job_id, nfs, s3,
            execute=execute, resume=resume, delete_nfs=delete_nfs,
            parallel=parallel,
            state_file=state_file, _state_data=state_data,
        )
        results.append(stats)

        totals["files_total"] += stats["files_found"]
        totals[upload_key] += stats.get(upload_key, 0)
        totals["files_skipped"] += stats["files_skipped"]
        totals[verify_key] += stats.get(verify_key, 0)
        totals["files_deleted"] += stats["files_deleted"]
        totals[bytes_key] += stats.get(bytes_key, 0)

        if stats["errors"]:
            totals["jobs_with_errors"] += 1
            for err in stats["errors"]:
                logger.error("  %s", err)
        else:
            totals["jobs_migrated"] += 1
            # C-19: Update DB if requested and job succeeded
            if update_db and execute:
                verified = stats.get("files_verified", 0)
                expected = stats["files_found"] - stats["files_skipped"]
                if verified >= expected > 0:
                    _update_job_db(job_id, db_url=db_url)

    elapsed = time.monotonic() - start
    totals["elapsed_seconds"] = round(elapsed, 1)

    return {"totals": totals, "jobs": results, "dry_run": not execute}


def _check_s3_connectivity(
    s3_endpoint: str,
    s3_bucket: str,
    s3_access_key: str,
    s3_secret_key: str,
    s3_region: str = "",
) -> str:
    """Test S3 connectivity and return status string.

    Returns one of: "ok", "bucket_not_found", "auth_failed", "unreachable",
    or "error: <detail>".
    """
    try:
        s3 = S3Backend(
            endpoint=s3_endpoint,
            bucket=s3_bucket,
            access_key=s3_access_key,
            secret_key=s3_secret_key,
            region=s3_region,
        )
        # Try to list zero objects — validates credentials + bucket
        if s3.client is None:
            return "error: S3 client initialization failed"
        s3.client.list_objects_v2(Bucket=s3_bucket, MaxKeys=1)
        return "ok"
    except Exception as exc:
        exc_str = str(exc).lower()
        if "nosuchbucket" in exc_str or ("bucket" in exc_str and "not" in exc_str):
            return "bucket_not_found"
        if "accessdenied" in exc_str or "403" in exc_str or "invalidaccesskeyid" in exc_str:
            return "auth_failed"
        if "endpointconnectionerror" in exc_str or "connectionrefused" in exc_str or "could not connect" in exc_str:
            return "unreachable"
        return f"error: {type(exc).__name__}: {str(exc)[:200]}"


def check_connectivity(
    nfs_root: str,
    s3_endpoint: str,
    s3_bucket: str,
    s3_access_key: str,
    s3_secret_key: str,
    s3_region: str = "",
) -> tuple[bool, list[str]]:
    """Pre-flight connectivity check for both NFS and S3.

    Returns ``(ok, errors)`` where *ok* is True when both NFS source and S3
    target are reachable, and *errors* is a list of human-readable failure
    descriptions (empty when *ok* is True).
    """
    errors: list[str] = []

    # --- NFS source ---
    if not os.path.isdir(nfs_root):
        errors.append(f"NFS root directory does not exist or is not readable: {nfs_root}")
    else:
        try:
            os.listdir(nfs_root)
        except OSError as exc:
            errors.append(f"NFS root directory is not readable: {nfs_root} ({exc})")

    # --- S3 target (HEAD bucket) ---
    s3_status = _check_s3_connectivity(
        s3_endpoint=s3_endpoint,
        s3_bucket=s3_bucket,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        s3_region=s3_region,
    )
    if s3_status != "ok":
        errors.append(f"S3 connectivity failed: {s3_status}")

    return (len(errors) == 0, errors)


def run_validation(
    nfs_root: str,
    s3_endpoint: str,
    s3_bucket: str,
    s3_access_key: str,
    s3_secret_key: str,
    s3_region: str = "",
    *,
    job_ids: list[str] | None = None,
    transfer_rate_bps: int = DEFAULT_TRANSFER_RATE_BPS,
    sample: int | None = None,
) -> MigrationReport:
    """Validate migration readiness without transferring data.

    Enumerates NFS files, computes SHA-256 checksums, simulates S3 key
    mapping, estimates transfer time, and checks S3 connectivity.

    When *sample* is a positive integer, only the first *sample* files per
    job directory are inspected.  The report includes the total file count
    alongside the sampled count so operators can gauge scope.
    """
    report = MigrationReport(
        nfs_root=nfs_root,
        s3_endpoint=s3_endpoint,
        s3_bucket=s3_bucket,
    )

    start = time.monotonic()

    # --- Discover jobs ---
    if job_ids:
        all_job_ids = job_ids
    else:
        all_job_ids = discover_jobs(nfs_root)
    report.jobs_found = len(all_job_ids)

    if not all_job_ids:
        report.warnings.append("No jobs found under NFS root")

    # --- Walk NFS tree and build manifest ---
    nfs = NFSBackend(root=nfs_root)
    files_skipped_by_sample = 0

    for job_id in all_job_ids:
        prefix = f"jobs/{job_id}"
        nfs_keys = nfs.list_objects(prefix)
        if not nfs_keys:
            report.warnings.append(f"Job {job_id} has no files")
            continue

        # Apply sample limit per job when requested
        if sample is not None and sample > 0:
            sampled_keys = nfs_keys[:sample]
            files_skipped_by_sample += max(0, len(nfs_keys) - sample)
        else:
            sampled_keys = nfs_keys

        for key in sampled_keys:
            local_path = nfs.to_absolute_path(key)
            try:
                file_size = os.path.getsize(local_path)
            except OSError as exc:
                report.warnings.append(f"Cannot stat {local_path}: {exc}")
                continue

            try:
                checksum = sha256_file(local_path)
            except OSError as exc:
                report.warnings.append(f"Cannot hash {local_path}: {exc}")
                checksum = ""

            entry = FileManifestEntry(
                nfs_path=local_path,
                s3_key=key,
                size_bytes=file_size,
                sha256=checksum,
            )
            report.file_manifest.append(entry)
            report.total_files += 1
            report.total_bytes += file_size

    # --- Estimate transfer time ---
    if transfer_rate_bps > 0 and report.total_bytes > 0:
        report.estimated_time_seconds = report.total_bytes / transfer_rate_bps
    else:
        report.estimated_time_seconds = 0.0

    # --- Check S3 connectivity ---
    report.connectivity_status = _check_s3_connectivity(
        s3_endpoint=s3_endpoint,
        s3_bucket=s3_bucket,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        s3_region=s3_region,
    )
    if report.connectivity_status != "ok":
        report.warnings.append(f"S3 connectivity issue: {report.connectivity_status}")

    if files_skipped_by_sample > 0:
        report.warnings.append(
            f"Sampling active: inspected {report.total_files} files, "
            f"skipped {files_skipped_by_sample} (--sample {sample})"
        )

    report.scan_elapsed_seconds = time.monotonic() - start
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate job artifacts from NFS to S3 storage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--nfs-root", required=True, help="NFS root directory (e.g. /shared)")
    parser.add_argument("--s3-endpoint", required=True, help="S3 endpoint URL")
    parser.add_argument("--s3-bucket", required=True, help="S3 bucket name")
    parser.add_argument("--s3-access-key", default="", help="S3 access key (prefer S3_ACCESS_KEY env var)")
    parser.add_argument("--s3-secret-key", default="", help="S3 secret key (prefer S3_SECRET_KEY env var)")
    parser.add_argument("--s3-region", default="", help="S3 region (optional)")
    parser.add_argument(
        "--execute", action="store_true",
        help="Execute migration (default is dry run)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip files already present in S3",
    )
    parser.add_argument(
        "--delete-nfs", action="store_true",
        help="Delete NFS files after verified upload",
    )
    parser.add_argument(
        "--job-ids", nargs="*",
        help="Migrate specific job IDs (default: all jobs)",
    )
    parser.add_argument(
        "--output", default="",
        help="Write JSON migration report to file",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Validate mode: enumerate files, compute checksums, check S3 connectivity (no data transferred)",
    )
    parser.add_argument(
        "--output-report", default="",
        help="Write validation report to file (JSON format, requires --validate)",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Sample N files per job during --validate (quick sanity check on large datasets)",
    )
    parser.add_argument(
        "--parallel", type=int, default=1,
        help="Number of parallel file uploads per job (default: 1, max: 32)",
    )
    parser.add_argument(
        "--state-file", default=None,
        help="Persist migration progress to this JSON file (enables crash recovery)",
    )
    parser.add_argument(
        "--update-db", action="store_true",
        help="Update coordinator DB storage_backend='s3' after successful job migration",
    )
    parser.add_argument(
        "--db-url", default=None,
        help="Database URL for --update-db (overrides DATABASE_URL env var)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # --- MIG3: Resolve S3 credentials (credential_manager -> env -> CLI fallback) ---
    s3_access_key = _get_credential("S3_ACCESS_KEY") or args.s3_access_key
    s3_secret_key = _get_credential("S3_SECRET_KEY") or args.s3_secret_key
    if not s3_access_key or not s3_secret_key:
        logger.error(
            "S3 credentials required via S3_ACCESS_KEY/S3_SECRET_KEY env vars "
            "or --s3-access-key/--s3-secret-key flags"
        )
        return 1

    # --- H1: Validate NFS root exists ---
    if not os.path.isdir(args.nfs_root):
        logger.error("NFS root does not exist: %s", args.nfs_root)
        return 1

    # --- MIG1: Validate job IDs against path traversal ---
    _JOB_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+$")
    if args.job_ids:
        for jid in args.job_ids:
            if not _JOB_ID_PATTERN.match(jid):
                logger.error(
                    "Invalid job ID (must be alphanumeric/dash/underscore/dot): %s",
                    jid,
                )
                return 1

    if args.output_report and not args.validate:
        logger.error("--output-report requires --validate")
        return 1

    if args.sample is not None and not args.validate:
        logger.error("--sample requires --validate")
        return 1

    if args.delete_nfs and not args.execute:
        logger.error("--delete-nfs requires --execute")
        return 1

    if args.update_db and not args.execute:
        logger.error("--update-db requires --execute")
        return 1

    if args.parallel < 1 or args.parallel > 32:
        logger.error("--parallel must be between 1 and 32")
        return 1

    # --- Connectivity pre-check (runs for all modes) ---
    conn_ok, conn_errors = check_connectivity(
        nfs_root=args.nfs_root,
        s3_endpoint=args.s3_endpoint,
        s3_bucket=args.s3_bucket,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        s3_region=args.s3_region,
    )
    if not conn_ok:
        for err in conn_errors:
            logger.error("Connectivity check failed: %s", err)
        return 1

    # --- Validate mode ---
    if args.validate:
        report = run_validation(
            nfs_root=args.nfs_root,
            s3_endpoint=args.s3_endpoint,
            s3_bucket=args.s3_bucket,
            s3_access_key=s3_access_key,
            s3_secret_key=s3_secret_key,
            s3_region=args.s3_region,
            job_ids=args.job_ids,
            sample=args.sample,
        )
        print(report.summary_text())

        if args.output_report:
            with open(args.output_report, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2)
            print(f"Validation report written to {args.output_report}")

        return 1 if report.connectivity_status != "ok" else 0

    # --- Dry-run banner ---
    is_dry_run = not args.execute
    if is_dry_run:
        print("=" * 60)
        print("=== DRY RUN MODE -- No files will be transferred ===")
        print("=" * 60)

    summary = run_migration(
        nfs_root=args.nfs_root,
        s3_endpoint=args.s3_endpoint,
        s3_bucket=args.s3_bucket,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        s3_region=args.s3_region,
        execute=args.execute,
        resume=args.resume,
        delete_nfs=args.delete_nfs,
        job_ids=args.job_ids,
        parallel=args.parallel,
        state_file=args.state_file,
        update_db=args.update_db,
        db_url=args.db_url,
    )

    t = summary["totals"]
    mode = "LIVE" if args.execute else "DRY RUN"
    print(f"\n{'='*60}")
    print(f"Migration Summary ({mode})")
    print(f"{'='*60}")
    print(f"  Jobs total:        {t['jobs_total']}")
    print(f"  Jobs migrated:     {t['jobs_migrated']}")
    print(f"  Jobs with errors:  {t['jobs_with_errors']}")

    if is_dry_run:
        print(f"  Files to upload (projected):   {t.get('files_to_upload', 0)}")
        print(f"  Files skipped:                 {t['files_skipped']}")
        print(f"  Files to verify (projected):   {t.get('files_to_verify', 0)}")
        print(f"  Files deleted:                 {t['files_deleted']}")
        print(f"  Bytes to transfer (projected): {t.get('bytes_to_upload', 0):,}")
    else:
        print(f"  Files uploaded:    {t.get('files_uploaded', 0)}")
        print(f"  Files skipped:     {t['files_skipped']}")
        print(f"  Files verified:    {t.get('files_verified', 0)}")
        print(f"  Files deleted:     {t['files_deleted']}")
        print(f"  Bytes transferred: {t.get('bytes_uploaded', 0):,}")
    print(f"  Elapsed:           {t['elapsed_seconds']}s")
    print(f"{'='*60}")

    if is_dry_run:
        print("=== DRY RUN COMPLETE -- No files were transferred ===")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Report written to {args.output}")

    return 1 if t["jobs_with_errors"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
