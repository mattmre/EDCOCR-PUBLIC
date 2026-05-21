#!/usr/bin/env python3
"""NFS-to-S3 migration validation module.

Verifies data integrity after migrating files from NFS storage to S3 by
comparing SHA-256 checksums of local files against S3-side digests.

Supports manifest-based workflows:
  1. Scan NFS directory, compute checksums, generate manifest.
  2. After migration, load manifest and validate against S3 checksums.
  3. Resume interrupted validations (only re-check PENDING/FAILED records).

Usage:
    # Generate a manifest of NFS files:
    python scripts/migration_validator.py --nfs-root /shared/data --generate-manifest

    # Validate against S3 checksums:
    python scripts/migration_validator.py --nfs-root /shared/data \\
        --validate --s3-checksums checksums.json

    # Resume a partially-completed validation:
    python scripts/migration_validator.py --manifest manifest.json \\
        --resume --s3-checksums checksums.json

    # Output report as JSON:
    python scripts/migration_validator.py --nfs-root /shared/data \\
        --validate --s3-checksums checksums.json --json
"""

from __future__ import annotations

import argparse
import enum
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field

logger = logging.getLogger("migration_validator")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MigrationStatus(enum.Enum):
    """Status of a single file in the migration validation pipeline."""

    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FileRecord:
    """Record of a single file tracked through migration validation."""

    path: str
    size_bytes: int
    sha256: str
    status: MigrationStatus = MigrationStatus.PENDING
    s3_key: str | None = None
    s3_etag: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict."""
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "status": self.status.value,
            "s3_key": self.s3_key,
            "s3_etag": self.s3_etag,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> FileRecord:
        """Deserialize from a dict (e.g. loaded from JSON manifest)."""
        return cls(
            path=data["path"],
            size_bytes=data["size_bytes"],
            sha256=data["sha256"],
            status=MigrationStatus(data["status"]),
            s3_key=data.get("s3_key"),
            s3_etag=data.get("s3_etag"),
            error=data.get("error"),
        )


@dataclass
class MigrationReport:
    """Aggregate report produced by a validation run."""

    total_files: int = 0
    verified: int = 0
    failed: int = 0
    skipped: int = 0
    pending: int = 0
    total_bytes: int = 0
    verified_bytes: int = 0
    duration_seconds: float = 0.0
    errors: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict."""
        return {
            "total_files": self.total_files,
            "verified": self.verified,
            "failed": self.failed,
            "skipped": self.skipped,
            "pending": self.pending,
            "total_bytes": self.total_bytes,
            "verified_bytes": self.verified_bytes,
            "duration_seconds": round(self.duration_seconds, 3),
            "errors": self.errors,
        }

    def summary_text(self) -> str:
        """Human-readable summary string."""
        lines = [
            "=" * 60,
            "Migration Validation Report",
            "=" * 60,
            f"  Total files:       {self.total_files}",
            f"  Verified:          {self.verified}",
            f"  Failed:            {self.failed}",
            f"  Skipped:           {self.skipped}",
            f"  Pending:           {self.pending}",
            f"  Total bytes:       {self.total_bytes}",
            f"  Verified bytes:    {self.verified_bytes}",
            f"  Duration:          {self.duration_seconds:.3f}s",
        ]
        if self.errors:
            lines.append(f"  Errors:            {len(self.errors)}")
            for err in self.errors:
                lines.append(f"    - {err.get('path', '?')}: {err.get('error', '?')}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class MigrationValidator:
    """Validates data integrity of NFS-to-S3 file migrations.

    Scans NFS directories, computes SHA-256 checksums, and compares them
    against S3-side checksums to verify migration correctness.
    """

    def __init__(
        self,
        nfs_root: str,
        s3_bucket: str = "",
        s3_prefix: str = "",
        manifest_path: str | None = None,
    ) -> None:
        self.nfs_root = nfs_root
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.manifest_path = manifest_path

    # -- Scanning -----------------------------------------------------------

    def scan_nfs(
        self,
        extensions: list[str] | None = None,
    ) -> list[FileRecord]:
        """Walk *nfs_root*, compute SHA-256 for each file, return records.

        Parameters
        ----------
        extensions:
            Optional whitelist of file extensions (e.g. ``[".pdf", ".txt"]``).
            When provided, only files whose extension (lowercased) is in this
            list are included.  Extensions should include the leading dot.

        Returns
        -------
        list[FileRecord]
            One record per discovered file, sorted by path.
        """
        records: list[FileRecord] = []
        norm_exts: set[str] | None = None
        if extensions is not None:
            norm_exts = {ext.lower() for ext in extensions}

        for dirpath, _dirnames, filenames in os.walk(self.nfs_root):
            for fname in sorted(filenames):
                full = os.path.join(dirpath, fname)
                if norm_exts is not None:
                    _, ext = os.path.splitext(fname)
                    if ext.lower() not in norm_exts:
                        continue
                try:
                    size = os.path.getsize(full)
                    digest = self.compute_sha256(full)
                    rel = os.path.relpath(full, self.nfs_root).replace("\\", "/")
                    s3_key = f"{self.s3_prefix}{rel}" if self.s3_prefix else rel
                    records.append(
                        FileRecord(
                            path=rel,
                            size_bytes=size,
                            sha256=digest,
                            s3_key=s3_key,
                        )
                    )
                except OSError as exc:
                    logger.warning("Skipping %s: %s", full, exc)

        records.sort(key=lambda r: r.path)
        return records

    # -- Hashing ------------------------------------------------------------

    @staticmethod
    def compute_sha256(file_path: str) -> str:
        """Return hex SHA-256 digest of a local file (chunk-based)."""
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    # -- Comparison ---------------------------------------------------------

    def compare_checksums(
        self,
        nfs_records: list[FileRecord],
        s3_checksums: dict[str, str],
    ) -> list[FileRecord]:
        """Compare NFS SHA-256 values against provided S3 checksums.

        Parameters
        ----------
        nfs_records:
            Records produced by :meth:`scan_nfs` (or loaded from manifest).
        s3_checksums:
            Mapping of ``s3_key -> sha256_hex`` as reported by the S3 side.

        Returns
        -------
        list[FileRecord]
            Updated records with ``status`` set to VERIFIED or FAILED.
        """
        for rec in nfs_records:
            lookup_key = rec.s3_key if rec.s3_key else rec.path
            s3_hash = s3_checksums.get(lookup_key)
            if s3_hash is None:
                rec.status = MigrationStatus.FAILED
                rec.error = f"Missing from S3 checksums: {lookup_key}"
            elif s3_hash == rec.sha256:
                rec.status = MigrationStatus.VERIFIED
                rec.s3_etag = s3_hash
                rec.error = None
            else:
                rec.status = MigrationStatus.FAILED
                rec.error = (
                    f"Checksum mismatch: NFS={rec.sha256} S3={s3_hash}"
                )
        return nfs_records

    # -- Manifest I/O -------------------------------------------------------

    def generate_manifest(
        self,
        records: list[FileRecord],
        output_path: str,
    ) -> None:
        """Write *records* to a JSON manifest file at *output_path*."""
        data = {
            "nfs_root": self.nfs_root,
            "s3_bucket": self.s3_bucket,
            "s3_prefix": self.s3_prefix,
            "file_count": len(records),
            "files": [r.to_dict() for r in records],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Manifest written to %s (%d records)", output_path, len(records))

    def load_manifest(self, manifest_path: str) -> list[FileRecord]:
        """Load records from a JSON manifest file."""
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [FileRecord.from_dict(entry) for entry in data["files"]]

    # -- Full validation ----------------------------------------------------

    def validate(
        self,
        nfs_records: list[FileRecord],
        s3_checksums: dict[str, str],
    ) -> MigrationReport:
        """Run full validation, returning a :class:`MigrationReport`.

        Compares each record's SHA-256 against the provided S3 checksums
        and produces aggregate statistics.
        """
        start = time.monotonic()

        self.compare_checksums(nfs_records, s3_checksums)

        verified = 0
        failed = 0
        skipped = 0
        pending = 0
        verified_bytes = 0
        total_bytes = 0
        errors: list[dict] = []

        for rec in nfs_records:
            total_bytes += rec.size_bytes
            if rec.status == MigrationStatus.VERIFIED:
                verified += 1
                verified_bytes += rec.size_bytes
            elif rec.status == MigrationStatus.FAILED:
                failed += 1
                errors.append({"path": rec.path, "error": rec.error or "unknown"})
            elif rec.status == MigrationStatus.SKIPPED:
                skipped += 1
            else:
                pending += 1

        elapsed = time.monotonic() - start

        return MigrationReport(
            total_files=len(nfs_records),
            verified=verified,
            failed=failed,
            skipped=skipped,
            pending=pending,
            total_bytes=total_bytes,
            verified_bytes=verified_bytes,
            duration_seconds=elapsed,
            errors=errors,
        )

    # -- Resume validation --------------------------------------------------

    def resume_validation(
        self,
        manifest_path: str,
        s3_checksums: dict[str, str],
    ) -> MigrationReport:
        """Load a manifest and re-validate only PENDING / FAILED records.

        Records already VERIFIED or SKIPPED are left untouched.
        """
        records = self.load_manifest(manifest_path)

        start = time.monotonic()

        to_check: list[FileRecord] = []
        already_done: list[FileRecord] = []

        for rec in records:
            if rec.status in (MigrationStatus.PENDING, MigrationStatus.FAILED):
                to_check.append(rec)
            else:
                already_done.append(rec)

        # Re-validate the subset
        self.compare_checksums(to_check, s3_checksums)

        # Aggregate over all records
        all_records = already_done + to_check

        verified = 0
        failed = 0
        skipped = 0
        pending = 0
        verified_bytes = 0
        total_bytes = 0
        errors: list[dict] = []

        for rec in all_records:
            total_bytes += rec.size_bytes
            if rec.status == MigrationStatus.VERIFIED:
                verified += 1
                verified_bytes += rec.size_bytes
            elif rec.status == MigrationStatus.FAILED:
                failed += 1
                errors.append({"path": rec.path, "error": rec.error or "unknown"})
            elif rec.status == MigrationStatus.SKIPPED:
                skipped += 1
            else:
                pending += 1

        elapsed = time.monotonic() - start

        return MigrationReport(
            total_files=len(all_records),
            verified=verified,
            failed=failed,
            skipped=skipped,
            pending=pending,
            total_bytes=total_bytes,
            verified_bytes=verified_bytes,
            duration_seconds=elapsed,
            errors=errors,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the migration validator CLI."""
    parser = argparse.ArgumentParser(
        description="NFS-to-S3 migration validation tool",
    )
    parser.add_argument(
        "--nfs-root",
        default="",
        help="Path to the NFS root directory to scan",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to an existing JSON manifest file",
    )
    parser.add_argument(
        "--generate-manifest",
        action="store_true",
        default=False,
        help="Scan NFS and write a manifest file (default: manifest.json)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=False,
        help="Validate NFS files against S3 checksums",
    )
    parser.add_argument(
        "--s3-checksums",
        default=None,
        help="Path to JSON file mapping s3_key -> sha256 hex digest",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output report as JSON instead of human-readable text",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume validation from manifest, re-checking PENDING/FAILED only",
    )
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=None,
        help="File extensions to include (e.g. .pdf .txt)",
    )
    parser.add_argument(
        "--s3-bucket",
        default="",
        help="S3 bucket name (recorded in manifest metadata)",
    )
    parser.add_argument(
        "--s3-prefix",
        default="",
        help="S3 key prefix to prepend to relative paths",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for manifest or report (overrides default)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.  Returns 0 on success, 1 on error."""
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    validator = MigrationValidator(
        nfs_root=args.nfs_root,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        manifest_path=args.manifest,
    )

    # -- Generate manifest --------------------------------------------------
    if args.generate_manifest:
        if not args.nfs_root:
            print("ERROR: --nfs-root is required for --generate-manifest", file=sys.stderr)
            return 1
        records = validator.scan_nfs(extensions=args.extensions)
        output = args.output or "manifest.json"
        validator.generate_manifest(records, output)
        print(f"Manifest written to {output} ({len(records)} files)")
        return 0

    # -- Resume validation --------------------------------------------------
    if args.resume:
        manifest = args.manifest
        if not manifest:
            print("ERROR: --manifest is required for --resume", file=sys.stderr)
            return 1
        if not args.s3_checksums:
            print("ERROR: --s3-checksums is required for --resume", file=sys.stderr)
            return 1
        with open(args.s3_checksums, "r", encoding="utf-8") as f:
            s3_checksums = json.load(f)
        report = validator.resume_validation(manifest, s3_checksums)
        _print_report(report, args.json, args.output)
        return 0 if report.failed == 0 else 1

    # -- Validate -----------------------------------------------------------
    if args.validate:
        if not args.s3_checksums:
            print("ERROR: --s3-checksums is required for --validate", file=sys.stderr)
            return 1
        with open(args.s3_checksums, "r", encoding="utf-8") as f:
            s3_checksums = json.load(f)

        if args.manifest:
            records = validator.load_manifest(args.manifest)
        else:
            if not args.nfs_root:
                print("ERROR: --nfs-root or --manifest is required for --validate", file=sys.stderr)
                return 1
            records = validator.scan_nfs(extensions=args.extensions)

        report = validator.validate(records, s3_checksums)
        _print_report(report, args.json, args.output)
        return 0 if report.failed == 0 else 1

    # -- No action specified ------------------------------------------------
    parser.print_help()
    return 1


def _print_report(report: MigrationReport, as_json: bool, output_path: str | None) -> None:
    """Print or write the validation report."""
    if as_json:
        text = json.dumps(report.to_dict(), indent=2)
    else:
        text = report.summary_text()

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")
        print(f"Report written to {output_path}")
    else:
        print(text)


if __name__ == "__main__":
    sys.exit(main())
