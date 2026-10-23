#!/usr/bin/env python3
"""Rehearse an object-level MinIO backup and restore inside an isolated lab.

Reads the source endpoint and credentials from Django coordinator settings,
exports the configured bucket to a temporary tar archive, restores current
object versions to an empty target bucket, and checks each restored SHA-256.
The archive can be deleted after the restore while retaining a metadata-only
evidence record. Restore credentials are read from environment variables; the
target endpoint is restricted to the dedicated ``ocr-closeout-c7`` restore
service so this canary cannot be pointed at an arbitrary S3 deployment.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import boto3
from botocore.exceptions import ClientError

RESTORE_TARGET_HOST = "ocr-kind-minio-restore.ocr-local.svc.cluster.local"
RESTORE_TARGET_PORT = 9000


def _client(endpoint: str, access_key: str, secret_key: str, region: str):
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region or "us-east-1",
    )


def _source_settings():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "coordinator.settings")
    import django

    django.setup()
    from django.conf import settings

    return settings


class _HashingReader:
    """Stream an S3 object into tar while calculating its content digest."""

    def __init__(self, body):
        self.body = body
        self.digest = hashlib.sha256()
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self.body.read(size)
        if chunk:
            self.digest.update(chunk)
            self.bytes_read += len(chunk)
        return chunk


def _hash_stream(stream) -> tuple[int, str]:
    digest = hashlib.sha256()
    byte_count = 0
    while chunk := stream.read(8 * 1024 * 1024):
        digest.update(chunk)
        byte_count += len(chunk)
    return byte_count, digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_target_endpoint(endpoint: str, source_endpoint: str) -> None:
    """Refuse arbitrary endpoints; this collector is limited to the named kind lab."""
    try:
        parsed = urlsplit(endpoint)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("restore target endpoint is malformed") from exc
    if (
        parsed.scheme != "http"
        or parsed.hostname != RESTORE_TARGET_HOST
        or port != RESTORE_TARGET_PORT
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "restore target must be the isolated kind MinIO restore service "
            f"http://{RESTORE_TARGET_HOST}:{RESTORE_TARGET_PORT}"
        )
    source = urlsplit(source_endpoint)
    if (source.scheme, source.hostname, source.port or 80) == (
        parsed.scheme,
        parsed.hostname,
        port,
    ):
        raise ValueError("source and restore target endpoints must be different")


def _endpoint_identity_for_evidence(endpoint: str) -> str:
    """Retain service origin while omitting URL credentials and path data."""
    try:
        parsed = urlsplit(endpoint)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return "invalid-endpoint"
    if not parsed.scheme or not hostname:
        return "invalid-endpoint"
    host = f"[{hostname}]" if ":" in hostname else hostname
    port_suffix = f":{port}" if port is not None else ""
    return f"{parsed.scheme}://{host}{port_suffix}"


def _validate_run_inputs(
    archive: Path,
    evidence: Path,
    max_objects: int,
    max_object_bytes: int,
    max_total_bytes: int,
) -> None:
    if min(max_objects, max_object_bytes, max_total_bytes) <= 0:
        raise ValueError("all restore sample limits must be positive")
    archive_resolved = archive.resolve(strict=False)
    evidence_resolved = evidence.resolve(strict=False)
    if archive_resolved == evidence_resolved:
        raise ValueError("archive and evidence paths must be different")
    if archive.exists() or archive.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing archive: {archive}")
    if evidence.exists() or evidence.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing evidence: {evidence}")


def _write_evidence(path: Path, evidence: dict[str, object]) -> None:
    """Atomically replace this run's marker with its final evidence record."""
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            output.write(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _export(
    source,
    bucket: str,
    archive: Path,
    max_total_bytes: int,
    max_objects: int,
    max_object_bytes: int,
):
    manifest: list[dict[str, object]] = []
    total_bytes = 0
    source_object_count = 0
    source_object_bytes = 0
    skipped_oversized = 0
    paginator = source.get_paginator("list_objects_v2")
    # Keep the local rehearsal archive uncompressed. Python's gzip-backed
    # tar reader emulates per-member seeks by replaying the compressed stream;
    # with thousands of large objects this multiplies read work dramatically.
    # Exclusive creation prevents a mistaken archive path from truncating an
    # existing file or following a pre-existing symlink.
    with tarfile.open(archive, mode="x") as tar:
        for page in paginator.paginate(Bucket=bucket):
            for item in page.get("Contents", []):
                listed_size = int(item["Size"])
                source_object_count += 1
                source_object_bytes += listed_size
                if listed_size > max_object_bytes:
                    skipped_oversized += 1
                    continue
                if len(manifest) >= max_objects:
                    continue
                key = str(item["Key"])
                response = source.get_object(Bucket=bucket, Key=key)
                expected_size = int(response["ContentLength"])
                if expected_size > max_object_bytes:
                    raise RuntimeError("source object grew beyond max-object-bytes")
                if expected_size > max_total_bytes - total_bytes:
                    raise RuntimeError("MinIO export exceeded max-total-bytes")
                reader = _HashingReader(response["Body"])
                if expected_size != int(item["Size"]):
                    raise RuntimeError("source object length changed during export")
                total_bytes += expected_size
                index = len(manifest)
                name = f"objects/{index:08d}"
                tar_info = tarfile.TarInfo(name)
                tar_info.size = expected_size
                tar.addfile(tar_info, reader)
                if reader.bytes_read != expected_size:
                    raise RuntimeError("source object ended before its stated size")
                manifest.append(
                    {
                        "bucket": bucket,
                        "key": key,
                        "archive_name": name,
                        "size": reader.bytes_read,
                        "sha256": reader.digest.hexdigest(),
                        "content_type": response.get("ContentType"),
                        "metadata": response.get("Metadata") or {},
                    }
                )
        serialized = json.dumps(manifest, sort_keys=True).encode("utf-8")
        info = tarfile.TarInfo("manifest.json")
        info.size = len(serialized)
        tar.addfile(info, io.BytesIO(serialized))
    if not manifest:
        raise RuntimeError(
            "MinIO bucket had no objects inside the selected sample bounds"
        )
    return (
        manifest,
        total_bytes,
        source_object_count,
        source_object_bytes,
        skipped_oversized,
    )


def _ensure_bucket(target, bucket: str) -> None:
    try:
        target.head_bucket(Bucket=bucket)
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code not in {"404", "NoSuchBucket", "NotFound"}:
            raise
        target.create_bucket(Bucket=bucket)


def _assert_empty_unversioned_bucket(target, bucket: str) -> None:
    versioning = target.get_bucket_versioning(Bucket=bucket)
    if versioning.get("Status") in {"Enabled", "Suspended"}:
        raise RuntimeError("restore target bucket must have versioning disabled")
    versions = target.list_object_versions(Bucket=bucket, MaxKeys=1)
    if versions.get("Versions") or versions.get("DeleteMarkers"):
        raise RuntimeError("restore target bucket has prior object versions")
    existing = target.list_objects_v2(Bucket=bucket, MaxKeys=1).get("KeyCount", 0)
    if existing:
        raise RuntimeError("restore target bucket is not empty")


def _rollback_restored_objects(target, bucket: str, keys: list[str]) -> None:
    """Remove only keys this invocation wrote after an unsuccessful restore."""
    for offset in range(0, len(keys), 1000):
        chunk = keys[offset : offset + 1000]
        result = target.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": key} for key in chunk], "Quiet": False},
        )
        errors = result.get("Errors") or []
        if errors:
            raise RuntimeError("restore rollback could not delete every written key")
    _assert_empty_unversioned_bucket(target, bucket)


def _restore(target, bucket: str, archive: Path, manifest):
    _ensure_bucket(target, bucket)
    _assert_empty_unversioned_bucket(target, bucket)

    verified = 0
    written_keys: list[str] = []
    try:
        with tarfile.open(archive, mode="r:") as tar:
            # Walk uncompressed tar members in order to avoid repeated seeks.
            for record in manifest:
                member = tar.next()
                if member is None or member.name != str(record["archive_name"]):
                    raise RuntimeError("backup archive object order/name mismatch")
                stream = tar.extractfile(member)
                if stream is None:
                    raise RuntimeError("backup archive object is missing")
                bucket_name = str(record["bucket"])
                key = str(record["key"])
                put_args = {
                    "Bucket": bucket_name,
                    "Key": key,
                    "Body": stream,
                    "ContentLength": int(record["size"]),
                    "Metadata": record.get("metadata") or {},
                }
                if record.get("content_type"):
                    put_args["ContentType"] = record["content_type"]
                written_keys.append(key)
                # Track before the request: a timeout can happen after S3 has
                # accepted the write but before the client receives its reply.
                target.put_object(**put_args)
                restored_stream = target.get_object(Bucket=bucket_name, Key=key)["Body"]
                restored_size, digest = _hash_stream(restored_stream)
                if restored_size != int(record["size"]) or digest != record["sha256"]:
                    raise RuntimeError("restored object does not match backup hash")
                verified += 1
            manifest_member = tar.next()
            if manifest_member is None or manifest_member.name != "manifest.json":
                raise RuntimeError("backup archive manifest is missing or out of order")
            manifest_stream = tar.extractfile(manifest_member)
            if manifest_stream is None or json.load(manifest_stream) != manifest:
                raise RuntimeError(
                    "backup archive manifest does not match restore plan"
                )
    except Exception:
        if written_keys:
            try:
                _rollback_restored_objects(target, bucket, written_keys)
            except Exception as cleanup_exc:
                raise RuntimeError(
                    "restore failed and rollback was incomplete; inspect the "
                    "isolated target bucket before retrying"
                ) from cleanup_exc
        raise
    return verified


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-endpoint", required=True)
    parser.add_argument(
        "--target-access-key-env", default="OCR_RESTORE_TARGET_ACCESS_KEY"
    )
    parser.add_argument(
        "--target-secret-key-env", default="OCR_RESTORE_TARGET_SECRET_KEY"
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--max-objects", type=int, default=32)
    parser.add_argument("--max-object-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--max-total-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--delete-archive", action="store_true")
    args = parser.parse_args()

    try:
        _validate_run_inputs(
            args.archive,
            args.evidence,
            args.max_objects,
            args.max_object_bytes,
            args.max_total_bytes,
        )
    except (FileExistsError, ValueError) as exc:
        parser.error(str(exc))

    target_access_key = os.environ.get(args.target_access_key_env)
    target_secret_key = os.environ.get(args.target_secret_key_env)
    if not target_access_key or not target_secret_key:
        parser.error(
            "restore credentials must be set in the environment variables "
            f"{args.target_access_key_env!r} and {args.target_secret_key_env!r}"
        )

    settings = _source_settings()
    source_endpoint = str(settings.S3_ENDPOINT)
    try:
        _validate_target_endpoint(args.target_endpoint, source_endpoint)
    except ValueError as exc:
        parser.error(str(exc))
    source_endpoint_evidence = _endpoint_identity_for_evidence(source_endpoint)
    target_endpoint_evidence = _endpoint_identity_for_evidence(args.target_endpoint)
    source_bucket = str(settings.S3_BUCKET)
    region = str(settings.S3_REGION or "us-east-1")
    source = _client(
        source_endpoint,
        str(settings.S3_ACCESS_KEY),
        str(settings.S3_SECRET_KEY),
        region,
    )
    target = _client(
        args.target_endpoint,
        target_access_key,
        target_secret_key,
        region,
    )
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    with args.evidence.open("x", encoding="utf-8") as output:
        output.write(
            json.dumps(
                {
                    "schema_version": "vtm_minio_backup_restore_live/v1",
                    "status": "IN_PROGRESS",
                    "observed_at": started_at,
                    "proof_tier": "isolated-single-host-kind-minio-object-restore",
                    "source_endpoint": source_endpoint_evidence,
                    "target_endpoint": target_endpoint_evidence,
                },
                sort_keys=True,
            )
            + "\n"
        )

    stage = "source_export"
    try:
        manifest, total_bytes, source_count, source_bytes, skipped_oversized = _export(
            source,
            source_bucket,
            args.archive,
            args.max_total_bytes,
            args.max_objects,
            args.max_object_bytes,
        )
        stage = "target_restore"
        verified = _restore(target, source_bucket, args.archive, manifest)
        stage = "archive_verification"
        archive_digest = _sha256_file(args.archive)
        if args.delete_archive:
            stage = "archive_cleanup"
            args.archive.unlink()
        archive_deleted = not args.archive.exists()
        complete_snapshot = (
            len(manifest) == source_count
            and total_bytes == source_bytes
            and skipped_oversized == 0
        )
        evidence = {
            "schema_version": "vtm_minio_backup_restore_live/v1",
            "status": "PASS",
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "proof_tier": "isolated-single-host-kind-minio-object-restore",
            "source_endpoint": source_endpoint_evidence,
            "target_endpoint": target_endpoint_evidence,
            "bucket_count": 1,
            "source_inventory_objects": source_count,
            "source_inventory_bytes": source_bytes,
            "object_count": len(manifest),
            "object_bytes": total_bytes,
            "complete_bucket_snapshot": complete_snapshot,
            "oversized_objects_skipped": skipped_oversized,
            "selection": {
                "kind": "complete-bucket"
                if complete_snapshot
                else "bounded-restore-sample",
                "max_objects": args.max_objects,
                "max_object_bytes": args.max_object_bytes,
                "selected_objects": len(manifest),
            },
            "archive_sha256": archive_digest,
            "restored_object_hashes_match": verified == len(manifest),
            "restored_object_count": verified,
            "archive_deleted_after_restore": args.delete_archive and archive_deleted,
            "assertions": {
                "backup_created": True,
                "target_bucket_was_empty": True,
                "target_bucket_versioning_disabled": True,
                "all_selected_object_contents_restored": verified == len(manifest),
                "selected_content_sha256_verified": verified == len(manifest),
                "archive_deletion_policy_satisfied": (
                    not args.delete_archive or archive_deleted
                ),
            },
            "scope_limit": (
                "This proves current object bytes and selected HTTP metadata for one "
                "local MinIO bucket. It does not cover point-in-time consistency, "
                "bucket policy, IAM identities, object versions, retention locks, "
                "production S3, or cross-site recovery."
            ),
        }
        stage = "evidence_write"
        _write_evidence(args.evidence, evidence)
    except Exception as exc:
        failure = {
            "schema_version": "vtm_minio_backup_restore_live/v1",
            "status": "FAIL",
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "proof_tier": "isolated-single-host-kind-minio-object-restore",
            "source_endpoint": source_endpoint_evidence,
            "target_endpoint": target_endpoint_evidence,
            "stage": stage,
            "error_type": type(exc).__name__,
            "error": f"{stage} failed; details omitted from retained evidence",
            "archive_retained": args.archive.exists(),
            "scope_limit": "A failed local restore is diagnostic evidence, not a production result.",
        }
        _write_evidence(args.evidence, failure)
        raise
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "bucket_count": evidence["bucket_count"],
                "object_count": evidence["object_count"],
                "object_bytes": evidence["object_bytes"],
                "complete_bucket_snapshot": evidence["complete_bucket_snapshot"],
                "restored_object_hashes_match": evidence[
                    "restored_object_hashes_match"
                ],
                "archive_deleted_after_restore": evidence[
                    "archive_deleted_after_restore"
                ],
                "evidence": str(args.evidence),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
