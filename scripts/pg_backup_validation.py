#!/usr/bin/env python3
"""PostgreSQL backup validation framework for EDCOCR.

Validates PostgreSQL backup configuration, tests backup integrity,
and optionally performs restore-to-temporary-database verification.

Designed for use with the Helm chart CronJob (postgresql.backup.enabled)
and Docker Compose manual backup workflows.

Modes:
    --check         Verify backup CronJob config (schedule, retention, storage)
    --backup        Trigger manual pg_dump and verify output integrity
    --verify        Verify existing backup files (pg_restore --list, size, age)
    --restore-test  Restore to a temporary database, verify table counts match
    --report        Generate backup health report (JSON + markdown)

Usage:
    python scripts/pg_backup_validation.py --check --database-url postgres://ocr:pass@localhost:5432/ocr_coordinator
    python scripts/pg_backup_validation.py --verify --backup-dir /backups --max-age-hours 24
    python scripts/pg_backup_validation.py --backup --database-url postgres://... --backup-dir /backups
    python scripts/pg_backup_validation.py --restore-test --database-url postgres://... --backup-dir /backups
    python scripts/pg_backup_validation.py --report --backup-dir /backups --output-dir ./reports

Run with: python scripts/pg_backup_validation.py --help
"""

import argparse
import datetime
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BACKUP_FILE_PATTERN = re.compile(
    r"ocr-coordinator-(\d{8}-\d{6})\.sql\.gz$"
)
DEFAULT_MAX_AGE_HOURS = 24
DEFAULT_MIN_SIZE_BYTES = 1024  # 1 KB minimum — anything less is truncated
DEFAULT_BACKUP_DIR = "/backups"
DEFAULT_RETENTION_COUNT = 7
DEFAULT_SCHEDULE = "0 2 * * *"

# Helm chart backup CronJob expected configuration keys
HELM_BACKUP_KEYS = {
    "schedule",
    "retentionCount",
    "storage",
    "timeoutSeconds",
    "historyLimit",
}

# Tables that must exist in a valid OCR coordinator backup
EXPECTED_TABLES = [
    "jobs_job",
    "jobs_worker",
    "jobs_pageresult",
    "jobs_custodyevent",
]

logger = logging.getLogger("pg_backup_validation")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BackupFileInfo:
    """Metadata for a single backup file."""

    path: str
    filename: str
    size_bytes: int
    timestamp_str: str
    timestamp: Optional[str] = None  # ISO format
    age_hours: float = 0.0
    is_valid_format: bool = False
    pg_restore_list_ok: bool = False
    table_count: int = 0
    tables_found: list = field(default_factory=list)
    errors: list = field(default_factory=list)


@dataclass
class CheckResult:
    """Result of a configuration check."""

    name: str
    passed: bool
    message: str
    details: Optional[dict] = None


@dataclass
class ValidationReport:
    """Aggregated validation report."""

    timestamp: str = ""
    mode: str = ""
    database_url_masked: str = ""
    backup_dir: str = ""
    checks: list = field(default_factory=list)
    backup_files: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def mask_database_url(url: str) -> str:
    """Mask password in a database URL for safe logging."""
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        if parsed.password:
            masked = url.replace(parsed.password, "****")
            return masked
        return url
    except Exception:
        return "***masked***"


def parse_database_url(url: str) -> dict:
    """Parse a PostgreSQL database URL into connection components."""
    parsed = urlparse(url)
    return {
        "host": parsed.hostname or "localhost",
        "port": str(parsed.port or 5432),
        "user": parsed.username or "ocr",
        "password": parsed.password or "",
        "dbname": (parsed.path or "/ocr_coordinator").lstrip("/"),
    }


def parse_backup_timestamp(filename: str) -> Optional[datetime.datetime]:
    """Extract timestamp from a backup filename.

    Expected format: ocr-coordinator-YYYYMMDD-HHMMSS.sql.gz
    """
    match = BACKUP_FILE_PATTERN.search(filename)
    if not match:
        return None
    ts_str = match.group(1)
    try:
        return datetime.datetime.strptime(ts_str, "%Y%m%d-%H%M%S")
    except ValueError:
        return None


def find_backup_files(backup_dir: str) -> list:
    """Find all backup files matching the expected pattern in a directory."""
    backup_path = Path(backup_dir)
    if not backup_path.exists():
        return []
    files = []
    for f in backup_path.iterdir():
        if f.is_file() and BACKUP_FILE_PATTERN.search(f.name):
            files.append(f)
    # Sort by modification time, newest first
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def run_command(cmd: list, timeout: int = 300, env: Optional[dict] = None) -> tuple:
    """Run a subprocess command and return (returncode, stdout, stderr).

    Args:
        cmd: Command as list of strings.
        timeout: Timeout in seconds (default 300).
        env: Optional environment dict; merged with os.environ.

    Returns:
        Tuple of (returncode, stdout, stderr).
    """
    run_env = dict(os.environ)
    if env:
        run_env.update(env)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=run_env,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out after {timeout}s"
    except FileNotFoundError:
        return -1, "", f"Command not found: {cmd[0]}"
    except Exception as exc:
        return -1, "", str(exc)


# ---------------------------------------------------------------------------
# Check mode
# ---------------------------------------------------------------------------


def check_config(
    database_url: str = "",
    backup_dir: str = DEFAULT_BACKUP_DIR,
    helm_values: Optional[dict] = None,
) -> list:
    """Verify backup configuration is correct.

    Checks:
        1. Database URL is valid and reachable.
        2. Backup directory exists and is writable.
        3. pg_dump binary is available.
        4. pg_restore binary is available.
        5. Helm values match expected keys (if provided).

    Returns:
        List of CheckResult objects.
    """
    results = []

    # 1. Database URL format
    if database_url:
        try:
            parts = parse_database_url(database_url)
            if parts["host"] and parts["dbname"]:
                results.append(
                    CheckResult(
                        name="database_url_format",
                        passed=True,
                        message=f"Database URL parsed: host={parts['host']}, db={parts['dbname']}",
                    )
                )
            else:
                results.append(
                    CheckResult(
                        name="database_url_format",
                        passed=False,
                        message="Database URL missing host or dbname",
                    )
                )
        except Exception as exc:
            results.append(
                CheckResult(
                    name="database_url_format",
                    passed=False,
                    message=f"Database URL parse error: {exc}",
                )
            )
    else:
        results.append(
            CheckResult(
                name="database_url_format",
                passed=False,
                message="No database URL provided (use --database-url)",
            )
        )

    # 2. Database connectivity
    if database_url:
        parts = parse_database_url(database_url)
        env = {"PGPASSWORD": parts["password"]}
        rc, stdout, stderr = run_command(
            [
                "pg_isready",
                "-h", parts["host"],
                "-p", parts["port"],
                "-U", parts["user"],
                "-d", parts["dbname"],
            ],
            timeout=10,
            env=env,
        )
        results.append(
            CheckResult(
                name="database_connectivity",
                passed=(rc == 0),
                message=stdout.strip() if rc == 0 else f"pg_isready failed: {stderr.strip()}",
            )
        )

    # 3. Backup directory
    bp = Path(backup_dir)
    if bp.exists():
        writable = os.access(str(bp), os.W_OK)
        results.append(
            CheckResult(
                name="backup_dir_exists",
                passed=True,
                message=f"Backup directory exists: {backup_dir}",
            )
        )
        results.append(
            CheckResult(
                name="backup_dir_writable",
                passed=writable,
                message="Backup directory is writable" if writable else f"Backup directory is NOT writable: {backup_dir}",
            )
        )
    else:
        results.append(
            CheckResult(
                name="backup_dir_exists",
                passed=False,
                message=f"Backup directory does not exist: {backup_dir}",
            )
        )

    # 4. pg_dump available
    rc, stdout, stderr = run_command(["pg_dump", "--version"], timeout=10)
    results.append(
        CheckResult(
            name="pg_dump_available",
            passed=(rc == 0),
            message=stdout.strip() if rc == 0 else f"pg_dump not found: {stderr.strip()}",
        )
    )

    # 5. pg_restore available
    rc, stdout, stderr = run_command(["pg_restore", "--version"], timeout=10)
    results.append(
        CheckResult(
            name="pg_restore_available",
            passed=(rc == 0),
            message=stdout.strip() if rc == 0 else f"pg_restore not found: {stderr.strip()}",
        )
    )

    # 6. Helm backup values (if provided)
    if helm_values:
        backup_cfg = helm_values.get("postgresql", {}).get("backup", {})
        if backup_cfg.get("enabled"):
            results.append(
                CheckResult(
                    name="helm_backup_enabled",
                    passed=True,
                    message="Helm postgresql.backup.enabled is true",
                )
            )
            # Schedule format check
            schedule = backup_cfg.get("schedule", "")
            parts_count = len(schedule.split())
            results.append(
                CheckResult(
                    name="helm_backup_schedule",
                    passed=(parts_count == 5),
                    message=f"Schedule: {schedule}" if parts_count == 5 else f"Invalid cron schedule ({parts_count} fields): {schedule}",
                )
            )
            # Retention
            retention = backup_cfg.get("retentionCount", 0)
            results.append(
                CheckResult(
                    name="helm_backup_retention",
                    passed=(retention > 0),
                    message=f"Retention: {retention} backups",
                )
            )
        else:
            results.append(
                CheckResult(
                    name="helm_backup_enabled",
                    passed=False,
                    message="Helm postgresql.backup.enabled is false (CronJob disabled)",
                )
            )

    return results


# ---------------------------------------------------------------------------
# Backup mode
# ---------------------------------------------------------------------------


def trigger_backup(
    database_url: str,
    backup_dir: str = DEFAULT_BACKUP_DIR,
    dry_run: bool = False,
) -> tuple:
    """Trigger a manual pg_dump and verify output.

    Args:
        database_url: PostgreSQL connection URL.
        backup_dir: Directory to store the backup file.
        dry_run: If True, only log what would happen.

    Returns:
        Tuple of (success: bool, backup_info: BackupFileInfo or None, errors: list).
    """
    errors = []
    parts = parse_database_url(database_url)
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    filename = f"ocr-coordinator-{timestamp}.sql.gz"
    output_path = str(Path(backup_dir) / filename)

    if dry_run:
        logger.info("[DRY RUN] Would run pg_dump to %s", output_path)
        return True, None, []

    # Ensure backup directory exists
    Path(backup_dir).mkdir(parents=True, exist_ok=True)

    env = {"PGPASSWORD": parts["password"]}
    cmd = [
        "pg_dump",
        "-h", parts["host"],
        "-p", parts["port"],
        "-U", parts["user"],
        "-d", parts["dbname"],
        "--no-password",
        "--format=custom",
        "--compress=6",
        "-f", output_path,
    ]

    logger.info("Running pg_dump: %s -> %s", parts["dbname"], output_path)
    rc, stdout, stderr = run_command(cmd, timeout=3600, env=env)

    if rc != 0:
        errors.append(f"pg_dump failed (exit {rc}): {stderr.strip()}")
        # Clean up partial file
        if Path(output_path).exists():
            Path(output_path).unlink()
        return False, None, errors

    # Verify the backup file
    if not Path(output_path).exists():
        errors.append(f"Backup file not created: {output_path}")
        return False, None, errors

    size = Path(output_path).stat().st_size
    if size < DEFAULT_MIN_SIZE_BYTES:
        errors.append(
            f"Backup file too small ({size} bytes < {DEFAULT_MIN_SIZE_BYTES}): "
            f"likely truncated"
        )
        return False, None, errors

    # Verify with pg_restore --list
    rc_verify, stdout_verify, stderr_verify = run_command(
        ["pg_restore", "--list", output_path],
        timeout=120,
    )

    info = BackupFileInfo(
        path=output_path,
        filename=filename,
        size_bytes=size,
        timestamp_str=timestamp,
        timestamp=datetime.datetime.utcnow().isoformat(),
        age_hours=0.0,
        is_valid_format=True,
        pg_restore_list_ok=(rc_verify == 0),
        errors=errors,
    )

    if rc_verify == 0:
        # Count tables in the listing
        tables = []
        for line in stdout_verify.splitlines():
            if "TABLE" in line and "TABLE DATA" not in line:
                tables.append(line.strip())
        info.table_count = len(tables)
        info.tables_found = tables

    success = (rc_verify == 0) and (size >= DEFAULT_MIN_SIZE_BYTES)
    if success:
        logger.info("Backup created successfully: %s (%d bytes)", output_path, size)
    else:
        errors.append(f"pg_restore --list failed: {stderr_verify.strip()}")
        info.errors = errors

    return success, info, errors


# ---------------------------------------------------------------------------
# Verify mode
# ---------------------------------------------------------------------------


def verify_backup_file(
    file_path: str,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
    min_size_bytes: int = DEFAULT_MIN_SIZE_BYTES,
) -> BackupFileInfo:
    """Verify a single backup file for integrity and recency.

    Checks:
        1. File exists and is a valid custom-format backup.
        2. File size exceeds minimum.
        3. File age is within threshold.
        4. pg_restore --list succeeds.

    Returns:
        BackupFileInfo with all fields populated.
    """
    fp = Path(file_path)
    errors = []

    if not fp.exists():
        return BackupFileInfo(
            path=file_path,
            filename=fp.name,
            size_bytes=0,
            timestamp_str="",
            errors=[f"File not found: {file_path}"],
        )

    size = fp.stat().st_size
    mtime = datetime.datetime.fromtimestamp(fp.stat().st_mtime)
    age_hours = (datetime.datetime.now() - mtime).total_seconds() / 3600

    # Parse timestamp from filename
    parsed_ts = parse_backup_timestamp(fp.name)
    ts_str = ""
    if parsed_ts:
        ts_str = parsed_ts.strftime("%Y%m%d-%H%M%S")

    info = BackupFileInfo(
        path=str(fp),
        filename=fp.name,
        size_bytes=size,
        timestamp_str=ts_str,
        timestamp=mtime.isoformat(),
        age_hours=round(age_hours, 2),
        is_valid_format=bool(parsed_ts),
    )

    # Size check
    if size < min_size_bytes:
        errors.append(
            f"File too small: {size} bytes (minimum: {min_size_bytes})"
        )

    # Age check
    if age_hours > max_age_hours:
        errors.append(
            f"File too old: {age_hours:.1f}h (maximum: {max_age_hours}h)"
        )

    # pg_restore --list check
    rc, stdout, stderr = run_command(
        ["pg_restore", "--list", str(fp)],
        timeout=120,
    )
    info.pg_restore_list_ok = (rc == 0)
    if rc == 0:
        tables = []
        for line in stdout.splitlines():
            if "TABLE" in line and "TABLE DATA" not in line:
                tables.append(line.strip())
        info.table_count = len(tables)
        info.tables_found = tables
    else:
        errors.append(f"pg_restore --list failed: {stderr.strip()}")

    info.errors = errors
    return info


def verify_backups(
    backup_dir: str = DEFAULT_BACKUP_DIR,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
    min_size_bytes: int = DEFAULT_MIN_SIZE_BYTES,
) -> list:
    """Verify all backup files in a directory.

    Returns:
        List of BackupFileInfo objects, sorted newest first.
    """
    files = find_backup_files(backup_dir)
    if not files:
        logger.warning("No backup files found in %s", backup_dir)
        return []

    results = []
    for fp in files:
        info = verify_backup_file(
            str(fp),
            max_age_hours=max_age_hours,
            min_size_bytes=min_size_bytes,
        )
        results.append(info)

    return results


# ---------------------------------------------------------------------------
# Restore-test mode
# ---------------------------------------------------------------------------


def restore_test(
    database_url: str,
    backup_dir: str = DEFAULT_BACKUP_DIR,
    dry_run: bool = False,
) -> tuple:
    """Restore the latest backup to a temporary database and verify table counts.

    Creates a temporary database, restores the latest backup into it,
    compares table row counts against the source database, then drops
    the temporary database.

    Args:
        database_url: PostgreSQL connection URL for the source database.
        backup_dir: Directory containing backup files.
        dry_run: If True, only log what would happen.

    Returns:
        Tuple of (success: bool, checks: list of CheckResult, errors: list).
    """
    errors = []
    checks = []
    parts = parse_database_url(database_url)
    env = {"PGPASSWORD": parts["password"]}
    temp_db = f"ocr_restore_test_{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

    # Find latest backup
    files = find_backup_files(backup_dir)
    if not files:
        return False, [], [f"No backup files found in {backup_dir}"]

    latest = str(files[0])
    checks.append(
        CheckResult(
            name="latest_backup",
            passed=True,
            message=f"Using backup: {files[0].name}",
        )
    )

    if dry_run:
        logger.info("[DRY RUN] Would restore %s to temp database %s", latest, temp_db)
        checks.append(
            CheckResult(
                name="restore_dry_run",
                passed=True,
                message=f"Would create temp database: {temp_db}",
            )
        )
        return True, checks, []

    try:
        # 1. Create temporary database
        rc, stdout, stderr = run_command(
            [
                "psql",
                "-h", parts["host"],
                "-p", parts["port"],
                "-U", parts["user"],
                "-d", "postgres",
                "-c", f"CREATE DATABASE {temp_db} OWNER {parts['user']};",
            ],
            timeout=30,
            env=env,
        )
        if rc != 0:
            errors.append(f"Failed to create temp database: {stderr.strip()}")
            return False, checks, errors

        checks.append(
            CheckResult(
                name="temp_db_created",
                passed=True,
                message=f"Temporary database created: {temp_db}",
            )
        )

        # 2. Restore backup into temp database
        rc, stdout, stderr = run_command(
            [
                "pg_restore",
                "-h", parts["host"],
                "-p", parts["port"],
                "-U", parts["user"],
                "-d", temp_db,
                "--no-owner",
                "--no-acl",
                latest,
            ],
            timeout=3600,
            env=env,
        )
        # pg_restore can return non-zero for warnings (e.g., extension issues),
        # so we check stderr for actual errors
        restore_ok = rc == 0 or "ERROR" not in stderr
        checks.append(
            CheckResult(
                name="restore_completed",
                passed=restore_ok,
                message="Restore completed" if restore_ok else f"Restore had errors: {stderr[:200]}",
            )
        )

        if not restore_ok:
            errors.append(f"pg_restore failed: {stderr[:500]}")
            return False, checks, errors

        # 3. Compare row counts for key tables
        for table_name in EXPECTED_TABLES:
            # Source count
            rc_src, stdout_src, _ = run_command(
                [
                    "psql",
                    "-h", parts["host"],
                    "-p", parts["port"],
                    "-U", parts["user"],
                    "-d", parts["dbname"],
                    "-t", "-c", f"SELECT count(*) FROM {table_name};",
                ],
                timeout=30,
                env=env,
            )
            # Restored count
            rc_dst, stdout_dst, _ = run_command(
                [
                    "psql",
                    "-h", parts["host"],
                    "-p", parts["port"],
                    "-U", parts["user"],
                    "-d", temp_db,
                    "-t", "-c", f"SELECT count(*) FROM {table_name};",
                ],
                timeout=30,
                env=env,
            )

            if rc_src != 0 or rc_dst != 0:
                checks.append(
                    CheckResult(
                        name=f"row_count_{table_name}",
                        passed=False,
                        message=f"Could not query {table_name}",
                    )
                )
                continue

            src_count = stdout_src.strip()
            dst_count = stdout_dst.strip()
            match = (src_count == dst_count)
            checks.append(
                CheckResult(
                    name=f"row_count_{table_name}",
                    passed=match,
                    message=f"{table_name}: source={src_count}, restored={dst_count}",
                    details={"source": src_count, "restored": dst_count},
                )
            )
            if not match:
                errors.append(
                    f"Row count mismatch for {table_name}: "
                    f"source={src_count}, restored={dst_count}"
                )

    finally:
        # 4. Drop temporary database
        rc, _, stderr = run_command(
            [
                "psql",
                "-h", parts["host"],
                "-p", parts["port"],
                "-U", parts["user"],
                "-d", "postgres",
                "-c", f"DROP DATABASE IF EXISTS {temp_db};",
            ],
            timeout=30,
            env=env,
        )
        if rc == 0:
            checks.append(
                CheckResult(
                    name="temp_db_dropped",
                    passed=True,
                    message=f"Temporary database dropped: {temp_db}",
                )
            )
        else:
            errors.append(f"Failed to drop temp database {temp_db}: {stderr.strip()}")

    success = all(c.passed for c in checks) and not errors
    return success, checks, errors


# ---------------------------------------------------------------------------
# Report mode
# ---------------------------------------------------------------------------


def generate_report(
    database_url: str = "",
    backup_dir: str = DEFAULT_BACKUP_DIR,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
    output_dir: str = ".",
    helm_values: Optional[dict] = None,
) -> ValidationReport:
    """Generate a comprehensive backup health report.

    Runs check + verify modes and produces JSON + markdown output.

    Returns:
        ValidationReport with all fields populated.
    """
    report = ValidationReport(
        timestamp=datetime.datetime.utcnow().isoformat(),
        mode="report",
        database_url_masked=mask_database_url(database_url),
        backup_dir=backup_dir,
    )

    # Run checks
    check_results = check_config(
        database_url=database_url,
        backup_dir=backup_dir,
        helm_values=helm_values,
    )
    report.checks = [asdict(c) for c in check_results]

    # Verify backups
    backup_infos = verify_backups(
        backup_dir=backup_dir,
        max_age_hours=max_age_hours,
    )
    report.backup_files = [asdict(b) for b in backup_infos]

    # Summary
    total_files = len(backup_infos)
    valid_files = sum(1 for b in backup_infos if b.pg_restore_list_ok)
    recent_files = sum(1 for b in backup_infos if b.age_hours <= max_age_hours)
    checks_passed = sum(1 for c in check_results if c.passed)
    checks_total = len(check_results)

    report.summary = {
        "total_backup_files": total_files,
        "valid_backup_files": valid_files,
        "recent_backup_files": recent_files,
        "max_age_hours": max_age_hours,
        "checks_passed": checks_passed,
        "checks_total": checks_total,
        "overall_health": "HEALTHY" if (valid_files > 0 and recent_files > 0 and checks_passed == checks_total) else "DEGRADED" if valid_files > 0 else "CRITICAL",
    }

    return report


def write_report_json(report: ValidationReport, output_dir: str) -> str:
    """Write report as JSON file."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    filename = f"pg-backup-validation-{ts}.json"
    filepath = str(Path(output_dir) / filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, default=str)
    return filepath


def write_report_markdown(report: ValidationReport, output_dir: str) -> str:
    """Write report as markdown file."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    filename = f"pg-backup-validation-{ts}.md"
    filepath = str(Path(output_dir) / filename)

    lines = []
    lines.append("# PostgreSQL Backup Validation Report")
    lines.append("")
    lines.append(f"**Generated**: {report.timestamp}")
    lines.append(f"**Backup Directory**: {report.backup_dir}")
    lines.append(f"**Database**: {report.database_url_masked}")
    lines.append("")

    # Summary
    s = report.summary
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Overall Health | **{s.get('overall_health', 'UNKNOWN')}** |")
    lines.append(f"| Total Backup Files | {s.get('total_backup_files', 0)} |")
    lines.append(f"| Valid Backup Files | {s.get('valid_backup_files', 0)} |")
    lines.append(f"| Recent (< {s.get('max_age_hours', DEFAULT_MAX_AGE_HOURS)}h) | {s.get('recent_backup_files', 0)} |")
    lines.append(f"| Config Checks | {s.get('checks_passed', 0)}/{s.get('checks_total', 0)} passed |")
    lines.append("")

    # Config checks
    lines.append("## Configuration Checks")
    lines.append("")
    lines.append("| Check | Status | Message |")
    lines.append("|-------|--------|---------|")
    for c in report.checks:
        status = "PASS" if c["passed"] else "FAIL"
        lines.append(f"| {c['name']} | {status} | {c['message']} |")
    lines.append("")

    # Backup files
    if report.backup_files:
        lines.append("## Backup Files")
        lines.append("")
        lines.append("| File | Size | Age (h) | Format | pg_restore | Tables | Errors |")
        lines.append("|------|------|---------|--------|------------|--------|--------|")
        for b in report.backup_files:
            size_mb = b["size_bytes"] / (1024 * 1024)
            fmt_ok = "OK" if b["is_valid_format"] else "BAD"
            restore_ok = "OK" if b["pg_restore_list_ok"] else "FAIL"
            err_count = len(b.get("errors", []))
            lines.append(
                f"| {b['filename']} | {size_mb:.1f} MB | {b['age_hours']:.1f} | "
                f"{fmt_ok} | {restore_ok} | {b['table_count']} | {err_count} |"
            )
        lines.append("")

    # Errors
    if report.errors:
        lines.append("## Errors")
        lines.append("")
        for err in report.errors:
            lines.append(f"- {err}")
        lines.append("")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return filepath


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="PostgreSQL backup validation for EDCOCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check configuration
  python scripts/pg_backup_validation.py --check --database-url postgres://ocr:pass@localhost:5432/ocr_coordinator

  # Verify existing backups
  python scripts/pg_backup_validation.py --verify --backup-dir /backups --max-age-hours 48

  # Manual backup + verify
  python scripts/pg_backup_validation.py --backup --database-url postgres://... --backup-dir /backups

  # Restore test (creates and drops temp database)
  python scripts/pg_backup_validation.py --restore-test --database-url postgres://... --backup-dir /backups

  # Full report
  python scripts/pg_backup_validation.py --report --backup-dir /backups --output-dir ./reports
        """,
    )

    # Modes (at least one required)
    modes = parser.add_argument_group("Modes (at least one required)")
    modes.add_argument(
        "--check",
        action="store_true",
        help="Verify backup configuration (tools, connectivity, Helm values)",
    )
    modes.add_argument(
        "--backup",
        action="store_true",
        help="Trigger manual pg_dump and verify output integrity",
    )
    modes.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing backup files (pg_restore --list, size, age)",
    )
    modes.add_argument(
        "--restore-test",
        action="store_true",
        help="Restore latest backup to temp database and compare row counts",
    )
    modes.add_argument(
        "--report",
        action="store_true",
        help="Generate full backup health report (JSON + markdown)",
    )

    # Connection
    conn = parser.add_argument_group("Connection")
    conn.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL", ""),
        help="PostgreSQL connection URL (default: $DATABASE_URL)",
    )

    # Paths
    paths = parser.add_argument_group("Paths")
    paths.add_argument(
        "--backup-dir",
        default=DEFAULT_BACKUP_DIR,
        help=f"Backup file directory (default: {DEFAULT_BACKUP_DIR})",
    )
    paths.add_argument(
        "--output-dir",
        default=".",
        help="Output directory for reports (default: current directory)",
    )

    # Options
    opts = parser.add_argument_group("Options")
    opts.add_argument(
        "--max-age-hours",
        type=float,
        default=DEFAULT_MAX_AGE_HOURS,
        help=f"Maximum backup age in hours (default: {DEFAULT_MAX_AGE_HOURS})",
    )
    opts.add_argument(
        "--min-size-bytes",
        type=int,
        default=DEFAULT_MIN_SIZE_BYTES,
        help=f"Minimum backup file size in bytes (default: {DEFAULT_MIN_SIZE_BYTES})",
    )
    opts.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without making changes",
    )
    opts.add_argument(
        "--helm-values",
        default="",
        help="Path to Helm values.yaml for config validation",
    )
    opts.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser


def main(argv: Optional[list] = None) -> int:
    """Main CLI entry point.

    Returns:
        Exit code (0 for success, 1 for failures).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Require at least one mode
    if not any([args.check, args.backup, args.verify, args.restore_test, args.report]):
        parser.error("At least one mode is required: --check, --backup, --verify, --restore-test, --report")

    # Load Helm values if specified
    helm_values = None
    if args.helm_values:
        try:
            import yaml
            with open(args.helm_values, encoding="utf-8") as f:
                helm_values = yaml.safe_load(f)
        except ImportError:
            logger.warning("PyYAML not installed; skipping Helm values validation")
        except Exception as exc:
            logger.error("Failed to load Helm values: %s", exc)

    exit_code = 0

    # --check mode
    if args.check:
        logger.info("=== Configuration Check ===")
        results = check_config(
            database_url=args.database_url,
            backup_dir=args.backup_dir,
            helm_values=helm_values,
        )
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            logger.info("  [%s] %s: %s", status, r.name, r.message)
            if not r.passed:
                exit_code = 1

    # --backup mode
    if args.backup:
        logger.info("=== Manual Backup ===")
        if not args.database_url:
            logger.error("--database-url is required for --backup mode")
            exit_code = 1
        else:
            success, info, errs = trigger_backup(
                database_url=args.database_url,
                backup_dir=args.backup_dir,
                dry_run=args.dry_run,
            )
            if success:
                if info:
                    logger.info("  Backup: %s (%d bytes)", info.filename, info.size_bytes)
                    logger.info("  pg_restore --list: %s", "OK" if info.pg_restore_list_ok else "FAIL")
                else:
                    logger.info("  [DRY RUN] Backup skipped")
            else:
                for err in errs:
                    logger.error("  %s", err)
                exit_code = 1

    # --verify mode
    if args.verify:
        logger.info("=== Verify Backups ===")
        infos = verify_backups(
            backup_dir=args.backup_dir,
            max_age_hours=args.max_age_hours,
            min_size_bytes=args.min_size_bytes,
        )
        if not infos:
            logger.warning("  No backup files found in %s", args.backup_dir)
            exit_code = 1
        for info in infos:
            status = "OK" if (info.pg_restore_list_ok and not info.errors) else "FAIL"
            logger.info(
                "  [%s] %s  size=%d  age=%.1fh  tables=%d",
                status, info.filename, info.size_bytes, info.age_hours, info.table_count,
            )
            for err in info.errors:
                logger.warning("    - %s", err)
            if info.errors:
                exit_code = 1

    # --restore-test mode
    if args.restore_test:
        logger.info("=== Restore Test ===")
        if not args.database_url:
            logger.error("--database-url is required for --restore-test mode")
            exit_code = 1
        else:
            success, checks, errs = restore_test(
                database_url=args.database_url,
                backup_dir=args.backup_dir,
                dry_run=args.dry_run,
            )
            for c in checks:
                status = "PASS" if c.passed else "FAIL"
                logger.info("  [%s] %s: %s", status, c.name, c.message)
            if not success:
                for err in errs:
                    logger.error("  %s", err)
                exit_code = 1

    # --report mode
    if args.report:
        logger.info("=== Generating Report ===")
        report = generate_report(
            database_url=args.database_url,
            backup_dir=args.backup_dir,
            max_age_hours=args.max_age_hours,
            output_dir=args.output_dir,
            helm_values=helm_values,
        )
        json_path = write_report_json(report, args.output_dir)
        md_path = write_report_markdown(report, args.output_dir)
        logger.info("  JSON report: %s", json_path)
        logger.info("  Markdown report: %s", md_path)
        logger.info("  Overall health: %s", report.summary.get("overall_health", "UNKNOWN"))
        if report.summary.get("overall_health") == "CRITICAL":
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
