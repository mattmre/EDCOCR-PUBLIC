"""Tests for scripts/migration_validator.py.

Validates the NFS-to-S3 migration validation module including the
MigrationStatus enum, FileRecord / MigrationReport dataclasses,
MigrationValidator class (scan_nfs, compute_sha256, compare_checksums,
manifest I/O, validate, resume_validation), and CLI argument parsing.
"""

import hashlib
import importlib
import json
import os
import sys

import pytest

# ---------------------------------------------------------------------------
# Import the module under test via importlib (it lives in scripts/)
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

_mod = importlib.import_module("migration_validator")
MigrationStatus = _mod.MigrationStatus
FileRecord = _mod.FileRecord
MigrationReport = _mod.MigrationReport
MigrationValidator = _mod.MigrationValidator
build_parser = _mod.build_parser
main = _mod.main


# ===========================================================================
# Helpers
# ===========================================================================

def _sha256(data: bytes) -> str:
    """Compute SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def nfs_tree(tmp_path):
    """Create a mock NFS directory with known files."""
    root = tmp_path / "nfs"
    sub = root / "subdir"
    sub.mkdir(parents=True)
    (root / "file_a.txt").write_bytes(b"alpha")
    (root / "file_b.pdf").write_bytes(b"bravo")
    (sub / "file_c.txt").write_bytes(b"charlie")
    return root


@pytest.fixture
def nfs_tree_str(nfs_tree):
    """String form of the NFS tree path."""
    return str(nfs_tree)


@pytest.fixture
def sample_records(nfs_tree_str):
    """Scan the NFS tree and return records."""
    v = MigrationValidator(nfs_root=nfs_tree_str)
    return v.scan_nfs()


@pytest.fixture
def matching_checksums(sample_records):
    """Build an s3_checksums dict that matches every record."""
    return {(r.s3_key or r.path): r.sha256 for r in sample_records}


@pytest.fixture
def mismatching_checksums(sample_records):
    """Build an s3_checksums dict where every hash is wrong."""
    return {(r.s3_key or r.path): "0" * 64 for r in sample_records}


# ===========================================================================
# MigrationStatus enum tests
# ===========================================================================


class TestMigrationStatus:
    """Validate the MigrationStatus enum."""

    def test_pending_value(self):
        assert MigrationStatus.PENDING.value == "PENDING"

    def test_in_progress_value(self):
        assert MigrationStatus.IN_PROGRESS.value == "IN_PROGRESS"

    def test_verified_value(self):
        assert MigrationStatus.VERIFIED.value == "VERIFIED"

    def test_failed_value(self):
        assert MigrationStatus.FAILED.value == "FAILED"

    def test_skipped_value(self):
        assert MigrationStatus.SKIPPED.value == "SKIPPED"

    def test_member_count(self):
        assert len(MigrationStatus) == 5

    def test_roundtrip_from_string(self):
        for member in MigrationStatus:
            assert MigrationStatus(member.value) is member


# ===========================================================================
# FileRecord tests
# ===========================================================================


class TestFileRecord:
    """Validate the FileRecord dataclass."""

    def test_construction_defaults(self):
        rec = FileRecord(path="a.txt", size_bytes=10, sha256="abc")
        assert rec.path == "a.txt"
        assert rec.size_bytes == 10
        assert rec.sha256 == "abc"
        assert rec.status == MigrationStatus.PENDING
        assert rec.s3_key is None
        assert rec.s3_etag is None
        assert rec.error is None

    def test_construction_all_fields(self):
        rec = FileRecord(
            path="b.pdf",
            size_bytes=99,
            sha256="def456",
            status=MigrationStatus.VERIFIED,
            s3_key="prefix/b.pdf",
            s3_etag="etag123",
            error=None,
        )
        assert rec.status == MigrationStatus.VERIFIED
        assert rec.s3_key == "prefix/b.pdf"
        assert rec.s3_etag == "etag123"

    def test_to_dict(self):
        rec = FileRecord(path="x", size_bytes=1, sha256="h", status=MigrationStatus.FAILED, error="oops")
        d = rec.to_dict()
        assert d["path"] == "x"
        assert d["size_bytes"] == 1
        assert d["sha256"] == "h"
        assert d["status"] == "FAILED"
        assert d["error"] == "oops"
        assert d["s3_key"] is None

    def test_from_dict_roundtrip(self):
        rec = FileRecord(
            path="f.txt", size_bytes=42, sha256="aaa",
            status=MigrationStatus.VERIFIED, s3_key="k", s3_etag="e",
        )
        d = rec.to_dict()
        restored = FileRecord.from_dict(d)
        assert restored.path == rec.path
        assert restored.size_bytes == rec.size_bytes
        assert restored.sha256 == rec.sha256
        assert restored.status == rec.status
        assert restored.s3_key == rec.s3_key
        assert restored.s3_etag == rec.s3_etag
        assert restored.error == rec.error

    def test_from_dict_with_error(self):
        d = {"path": "z", "size_bytes": 0, "sha256": "", "status": "FAILED", "error": "bad"}
        rec = FileRecord.from_dict(d)
        assert rec.status == MigrationStatus.FAILED
        assert rec.error == "bad"


# ===========================================================================
# MigrationReport tests
# ===========================================================================


class TestMigrationReport:
    """Validate the MigrationReport dataclass."""

    def test_default_construction(self):
        report = MigrationReport()
        assert report.total_files == 0
        assert report.verified == 0
        assert report.failed == 0
        assert report.skipped == 0
        assert report.pending == 0
        assert report.total_bytes == 0
        assert report.verified_bytes == 0
        assert report.duration_seconds == 0.0
        assert report.errors == []

    def test_construction_with_values(self):
        report = MigrationReport(
            total_files=10, verified=8, failed=1, skipped=1,
            pending=0, total_bytes=1000, verified_bytes=800,
            duration_seconds=1.5,
            errors=[{"path": "x", "error": "bad"}],
        )
        assert report.total_files == 10
        assert report.verified == 8
        assert report.failed == 1
        assert report.skipped == 1
        assert report.verified_bytes == 800
        assert len(report.errors) == 1

    def test_to_dict(self):
        report = MigrationReport(total_files=3, verified=2, failed=1, total_bytes=500)
        d = report.to_dict()
        assert d["total_files"] == 3
        assert d["verified"] == 2
        assert d["failed"] == 1
        assert d["total_bytes"] == 500
        assert "duration_seconds" in d
        assert "errors" in d

    def test_to_dict_json_serializable(self):
        report = MigrationReport(
            total_files=1, errors=[{"path": "a", "error": "e"}],
        )
        text = json.dumps(report.to_dict())
        assert isinstance(text, str)

    def test_independent_instances_no_shared_state(self):
        r1 = MigrationReport()
        r2 = MigrationReport()
        r1.errors.append({"path": "x", "error": "y"})
        assert r2.errors == []

    def test_summary_text_contains_key_fields(self):
        report = MigrationReport(total_files=5, verified=3, failed=2, total_bytes=1024)
        text = report.summary_text()
        assert "Migration Validation Report" in text
        assert "5" in text
        assert "3" in text
        assert "2" in text
        assert "1024" in text

    def test_summary_text_shows_errors(self):
        report = MigrationReport(
            errors=[{"path": "bad.txt", "error": "checksum mismatch"}],
        )
        text = report.summary_text()
        assert "bad.txt" in text
        assert "checksum mismatch" in text

    def test_summary_text_no_errors(self):
        report = MigrationReport()
        text = report.summary_text()
        assert "Errors:" not in text


# ===========================================================================
# MigrationValidator construction tests
# ===========================================================================


class TestMigrationValidatorConstruction:
    """Validate MigrationValidator construction."""

    def test_default_construction(self, tmp_path):
        v = MigrationValidator(nfs_root=str(tmp_path))
        assert v.nfs_root == str(tmp_path)
        assert v.s3_bucket == ""
        assert v.s3_prefix == ""
        assert v.manifest_path is None

    def test_full_construction(self, tmp_path):
        v = MigrationValidator(
            nfs_root=str(tmp_path),
            s3_bucket="my-bucket",
            s3_prefix="data/",
            manifest_path="/tmp/m.json",
        )
        assert v.s3_bucket == "my-bucket"
        assert v.s3_prefix == "data/"
        assert v.manifest_path == "/tmp/m.json"


# ===========================================================================
# compute_sha256 tests
# ===========================================================================


class TestComputeSha256:
    """Validate SHA-256 file hashing."""

    def test_known_content(self, tmp_path):
        f = tmp_path / "hello.bin"
        f.write_bytes(b"hello")
        digest = MigrationValidator.compute_sha256(str(f))
        expected = _sha256(b"hello")
        assert digest == expected
        assert len(digest) == 64

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        digest = MigrationValidator.compute_sha256(str(f))
        expected = _sha256(b"")
        assert digest == expected

    def test_large_file(self, tmp_path):
        """Hash of a file larger than a single 8192-byte chunk."""
        data = b"X" * 20000
        f = tmp_path / "large.bin"
        f.write_bytes(data)
        digest = MigrationValidator.compute_sha256(str(f))
        assert digest == _sha256(data)

    def test_binary_content(self, tmp_path):
        data = bytes(range(256))
        f = tmp_path / "binary.bin"
        f.write_bytes(data)
        assert MigrationValidator.compute_sha256(str(f)) == _sha256(data)


# ===========================================================================
# scan_nfs tests
# ===========================================================================


class TestScanNfs:
    """Validate NFS directory scanning."""

    def test_finds_all_files(self, nfs_tree_str):
        v = MigrationValidator(nfs_root=nfs_tree_str)
        records = v.scan_nfs()
        assert len(records) == 3

    def test_records_have_correct_sha256(self, nfs_tree, nfs_tree_str):
        v = MigrationValidator(nfs_root=nfs_tree_str)
        records = v.scan_nfs()
        for rec in records:
            full = os.path.join(nfs_tree_str, rec.path.replace("/", os.sep))
            with open(full, "rb") as f:
                expected = _sha256(f.read())
            assert rec.sha256 == expected

    def test_records_have_correct_sizes(self, nfs_tree, nfs_tree_str):
        v = MigrationValidator(nfs_root=nfs_tree_str)
        records = v.scan_nfs()
        sizes = {r.path: r.size_bytes for r in records}
        assert sizes["file_a.txt"] == len(b"alpha")
        assert sizes["file_b.pdf"] == len(b"bravo")

    def test_paths_use_forward_slashes(self, nfs_tree_str):
        v = MigrationValidator(nfs_root=nfs_tree_str)
        records = v.scan_nfs()
        for rec in records:
            assert "\\" not in rec.path

    def test_records_sorted_by_path(self, nfs_tree_str):
        v = MigrationValidator(nfs_root=nfs_tree_str)
        records = v.scan_nfs()
        paths = [r.path for r in records]
        assert paths == sorted(paths)

    def test_status_is_pending(self, nfs_tree_str):
        v = MigrationValidator(nfs_root=nfs_tree_str)
        records = v.scan_nfs()
        for rec in records:
            assert rec.status == MigrationStatus.PENDING

    def test_s3_key_set(self, nfs_tree_str):
        v = MigrationValidator(nfs_root=nfs_tree_str)
        records = v.scan_nfs()
        for rec in records:
            assert rec.s3_key == rec.path  # no prefix

    def test_s3_key_with_prefix(self, nfs_tree_str):
        v = MigrationValidator(nfs_root=nfs_tree_str, s3_prefix="data/")
        records = v.scan_nfs()
        for rec in records:
            assert rec.s3_key.startswith("data/")
            assert rec.s3_key == f"data/{rec.path}"

    def test_empty_directory(self, tmp_path):
        scan_dir = tmp_path / "scan"
        scan_dir.mkdir()
        v = MigrationValidator(nfs_root=str(scan_dir))
        records = v.scan_nfs()
        assert records == []

    def test_extensions_filter_txt(self, nfs_tree_str):
        v = MigrationValidator(nfs_root=nfs_tree_str)
        records = v.scan_nfs(extensions=[".txt"])
        assert len(records) == 2
        for rec in records:
            assert rec.path.endswith(".txt")

    def test_extensions_filter_pdf(self, nfs_tree_str):
        v = MigrationValidator(nfs_root=nfs_tree_str)
        records = v.scan_nfs(extensions=[".pdf"])
        assert len(records) == 1
        assert records[0].path.endswith(".pdf")

    def test_extensions_filter_case_insensitive(self, tmp_path):
        (tmp_path / "doc.PDF").write_bytes(b"data")
        v = MigrationValidator(nfs_root=str(tmp_path))
        records = v.scan_nfs(extensions=[".pdf"])
        assert len(records) == 1

    def test_extensions_filter_no_match(self, nfs_tree_str):
        v = MigrationValidator(nfs_root=nfs_tree_str)
        records = v.scan_nfs(extensions=[".docx"])
        assert records == []

    def test_extensions_filter_multiple(self, nfs_tree_str):
        v = MigrationValidator(nfs_root=nfs_tree_str)
        records = v.scan_nfs(extensions=[".txt", ".pdf"])
        assert len(records) == 3


# ===========================================================================
# compare_checksums tests
# ===========================================================================


class TestCompareChecksums:
    """Validate checksum comparison logic."""

    def test_all_matching(self, nfs_tree_str, sample_records, matching_checksums):
        v = MigrationValidator(nfs_root=nfs_tree_str)
        updated = v.compare_checksums(sample_records, matching_checksums)
        for rec in updated:
            assert rec.status == MigrationStatus.VERIFIED
            assert rec.error is None

    def test_all_mismatching(self, nfs_tree_str, sample_records, mismatching_checksums):
        v = MigrationValidator(nfs_root=nfs_tree_str)
        updated = v.compare_checksums(sample_records, mismatching_checksums)
        for rec in updated:
            assert rec.status == MigrationStatus.FAILED
            assert "mismatch" in rec.error.lower()

    def test_missing_from_s3(self, nfs_tree_str, sample_records):
        v = MigrationValidator(nfs_root=nfs_tree_str)
        updated = v.compare_checksums(sample_records, {})
        for rec in updated:
            assert rec.status == MigrationStatus.FAILED
            assert "Missing" in rec.error

    def test_partial_match(self, nfs_tree_str, sample_records, matching_checksums):
        # Make one record fail by removing its checksum
        first_key = sample_records[0].s3_key or sample_records[0].path
        del matching_checksums[first_key]
        v = MigrationValidator(nfs_root=nfs_tree_str)
        updated = v.compare_checksums(sample_records, matching_checksums)
        statuses = [r.status for r in updated]
        assert MigrationStatus.FAILED in statuses
        assert MigrationStatus.VERIFIED in statuses

    def test_verified_sets_etag(self, nfs_tree_str, sample_records, matching_checksums):
        v = MigrationValidator(nfs_root=nfs_tree_str)
        updated = v.compare_checksums(sample_records, matching_checksums)
        for rec in updated:
            assert rec.s3_etag == rec.sha256

    def test_failed_preserves_none_etag(self, nfs_tree_str, sample_records, mismatching_checksums):
        v = MigrationValidator(nfs_root=nfs_tree_str)
        updated = v.compare_checksums(sample_records, mismatching_checksums)
        for rec in updated:
            assert rec.s3_etag is None


# ===========================================================================
# generate_manifest + load_manifest roundtrip tests
# ===========================================================================


class TestManifestRoundtrip:
    """Validate manifest generation and loading."""

    def test_roundtrip(self, nfs_tree_str, sample_records, tmp_path):
        manifest = str(tmp_path / "manifest.json")
        v = MigrationValidator(nfs_root=nfs_tree_str, s3_bucket="bucket")
        v.generate_manifest(sample_records, manifest)
        loaded = v.load_manifest(manifest)
        assert len(loaded) == len(sample_records)
        for orig, rest in zip(sample_records, loaded):
            assert orig.path == rest.path
            assert orig.size_bytes == rest.size_bytes
            assert orig.sha256 == rest.sha256
            assert orig.status == rest.status

    def test_manifest_is_valid_json(self, nfs_tree_str, sample_records, tmp_path):
        manifest = str(tmp_path / "manifest.json")
        v = MigrationValidator(nfs_root=nfs_tree_str)
        v.generate_manifest(sample_records, manifest)
        with open(manifest) as f:
            data = json.load(f)
        assert data["nfs_root"] == nfs_tree_str
        assert data["file_count"] == len(sample_records)
        assert "files" in data

    def test_manifest_preserves_status(self, nfs_tree_str, tmp_path):
        v = MigrationValidator(nfs_root=nfs_tree_str)
        records = v.scan_nfs()
        records[0].status = MigrationStatus.VERIFIED
        records[1].status = MigrationStatus.FAILED
        records[1].error = "oops"
        manifest = str(tmp_path / "manifest.json")
        v.generate_manifest(records, manifest)
        loaded = v.load_manifest(manifest)
        assert loaded[0].status == MigrationStatus.VERIFIED
        assert loaded[1].status == MigrationStatus.FAILED
        assert loaded[1].error == "oops"

    def test_manifest_includes_metadata(self, nfs_tree_str, sample_records, tmp_path):
        manifest = str(tmp_path / "manifest.json")
        v = MigrationValidator(nfs_root=nfs_tree_str, s3_bucket="b", s3_prefix="p/")
        v.generate_manifest(sample_records, manifest)
        with open(manifest) as f:
            data = json.load(f)
        assert data["s3_bucket"] == "b"
        assert data["s3_prefix"] == "p/"


# ===========================================================================
# validate tests
# ===========================================================================


class TestValidate:
    """Validate the full validation workflow."""

    def test_all_matching(self, nfs_tree_str, sample_records, matching_checksums):
        v = MigrationValidator(nfs_root=nfs_tree_str)
        report = v.validate(sample_records, matching_checksums)
        assert report.total_files == 3
        assert report.verified == 3
        assert report.failed == 0
        assert report.errors == []
        assert report.verified_bytes == report.total_bytes

    def test_all_failing(self, nfs_tree_str, sample_records, mismatching_checksums):
        v = MigrationValidator(nfs_root=nfs_tree_str)
        report = v.validate(sample_records, mismatching_checksums)
        assert report.total_files == 3
        assert report.verified == 0
        assert report.failed == 3
        assert len(report.errors) == 3
        assert report.verified_bytes == 0

    def test_partial_failure(self, nfs_tree_str, sample_records, matching_checksums):
        first_key = sample_records[0].s3_key or sample_records[0].path
        matching_checksums[first_key] = "0" * 64
        v = MigrationValidator(nfs_root=nfs_tree_str)
        report = v.validate(sample_records, matching_checksums)
        assert report.verified == 2
        assert report.failed == 1
        assert len(report.errors) == 1

    def test_missing_s3_keys(self, nfs_tree_str, sample_records):
        v = MigrationValidator(nfs_root=nfs_tree_str)
        report = v.validate(sample_records, {})
        assert report.failed == 3

    def test_duration_positive(self, nfs_tree_str, sample_records, matching_checksums):
        v = MigrationValidator(nfs_root=nfs_tree_str)
        report = v.validate(sample_records, matching_checksums)
        assert report.duration_seconds >= 0

    def test_total_bytes_correct(self, nfs_tree_str, sample_records, matching_checksums):
        v = MigrationValidator(nfs_root=nfs_tree_str)
        report = v.validate(sample_records, matching_checksums)
        expected = sum(r.size_bytes for r in sample_records)
        assert report.total_bytes == expected

    def test_error_dicts_have_path_and_error(self, nfs_tree_str, sample_records, mismatching_checksums):
        v = MigrationValidator(nfs_root=nfs_tree_str)
        report = v.validate(sample_records, mismatching_checksums)
        for err in report.errors:
            assert "path" in err
            assert "error" in err


# ===========================================================================
# resume_validation tests
# ===========================================================================


class TestResumeValidation:
    """Validate the resume validation workflow."""

    def test_skips_verified_records(self, nfs_tree_str, tmp_path):
        v = MigrationValidator(nfs_root=nfs_tree_str)
        records = v.scan_nfs()
        checksums = {(r.s3_key or r.path): r.sha256 for r in records}

        # Pre-verify one record
        records[0].status = MigrationStatus.VERIFIED

        manifest = str(tmp_path / "manifest.json")
        v.generate_manifest(records, manifest)

        # Corrupt the checksum for the verified record so that IF it were
        # re-checked it would fail.  Resume should skip it.
        first_key = records[0].s3_key or records[0].path
        checksums[first_key] = "0" * 64

        report = v.resume_validation(manifest, checksums)
        # The already-verified record should still count as verified
        assert report.verified >= 1
        assert report.total_files == 3

    def test_rechecks_pending_records(self, nfs_tree_str, tmp_path, matching_checksums):
        v = MigrationValidator(nfs_root=nfs_tree_str)
        records = v.scan_nfs()

        manifest = str(tmp_path / "manifest.json")
        v.generate_manifest(records, manifest)

        report = v.resume_validation(manifest, matching_checksums)
        assert report.verified == 3

    def test_rechecks_failed_records(self, nfs_tree_str, tmp_path, matching_checksums):
        v = MigrationValidator(nfs_root=nfs_tree_str)
        records = v.scan_nfs()
        records[0].status = MigrationStatus.FAILED
        records[0].error = "previous failure"

        manifest = str(tmp_path / "manifest.json")
        v.generate_manifest(records, manifest)

        # Now provide correct checksums — the FAILED record should be re-verified
        report = v.resume_validation(manifest, matching_checksums)
        assert report.verified == 3
        assert report.failed == 0

    def test_preserves_skipped_records(self, nfs_tree_str, tmp_path, matching_checksums):
        v = MigrationValidator(nfs_root=nfs_tree_str)
        records = v.scan_nfs()
        records[0].status = MigrationStatus.SKIPPED

        manifest = str(tmp_path / "manifest.json")
        v.generate_manifest(records, manifest)

        report = v.resume_validation(manifest, matching_checksums)
        assert report.skipped == 1
        assert report.verified == 2


# ===========================================================================
# CLI argument parsing tests
# ===========================================================================


class TestCLIParsing:
    """Validate CLI argument parsing via build_parser."""

    def test_nfs_root_arg(self):
        parser = build_parser()
        args = parser.parse_args(["--nfs-root", "/data"])
        assert args.nfs_root == "/data"

    def test_manifest_arg(self):
        parser = build_parser()
        args = parser.parse_args(["--manifest", "m.json"])
        assert args.manifest == "m.json"

    def test_generate_manifest_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--generate-manifest", "--nfs-root", "/data"])
        assert args.generate_manifest is True

    def test_validate_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--validate", "--s3-checksums", "c.json"])
        assert args.validate is True

    def test_s3_checksums_arg(self):
        parser = build_parser()
        args = parser.parse_args(["--s3-checksums", "sums.json"])
        assert args.s3_checksums == "sums.json"

    def test_json_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--json"])
        assert args.json is True

    def test_resume_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--resume", "--manifest", "m.json", "--s3-checksums", "c.json"])
        assert args.resume is True

    def test_extensions_arg(self):
        parser = build_parser()
        args = parser.parse_args(["--extensions", ".pdf", ".txt"])
        assert args.extensions == [".pdf", ".txt"]

    def test_s3_bucket_arg(self):
        parser = build_parser()
        args = parser.parse_args(["--s3-bucket", "my-bucket"])
        assert args.s3_bucket == "my-bucket"

    def test_s3_prefix_arg(self):
        parser = build_parser()
        args = parser.parse_args(["--s3-prefix", "prefix/"])
        assert args.s3_prefix == "prefix/"

    def test_output_arg(self):
        parser = build_parser()
        args = parser.parse_args(["--output", "out.json"])
        assert args.output == "out.json"

    def test_defaults(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.nfs_root == ""
        assert args.manifest is None
        assert args.generate_manifest is False
        assert args.validate is False
        assert args.s3_checksums is None
        assert args.json is False
        assert args.resume is False
        assert args.extensions is None
        assert args.s3_bucket == ""
        assert args.s3_prefix == ""
        assert args.output is None


# ===========================================================================
# CLI main() integration tests
# ===========================================================================


class TestCLIMain:
    """Validate the main() CLI entry point."""

    def test_no_args_returns_one(self):
        result = main([])
        assert result == 1

    def test_generate_manifest_writes_file(self, nfs_tree_str, tmp_path):
        out = str(tmp_path / "manifest.json")
        result = main([
            "--nfs-root", nfs_tree_str,
            "--generate-manifest",
            "--output", out,
        ])
        assert result == 0
        assert os.path.exists(out)
        with open(out) as f:
            data = json.load(f)
        assert data["file_count"] == 3

    def test_generate_manifest_without_nfs_root_fails(self):
        result = main(["--generate-manifest"])
        assert result == 1

    def test_validate_all_matching(self, nfs_tree_str, tmp_path):
        # First generate manifest
        manifest = str(tmp_path / "manifest.json")
        main(["--nfs-root", nfs_tree_str, "--generate-manifest", "--output", manifest])

        # Build matching checksums file
        with open(manifest) as f:
            data = json.load(f)
        checksums = {entry["s3_key"] or entry["path"]: entry["sha256"] for entry in data["files"]}
        checksums_file = str(tmp_path / "checksums.json")
        with open(checksums_file, "w") as f:
            json.dump(checksums, f)

        result = main([
            "--nfs-root", nfs_tree_str,
            "--validate",
            "--s3-checksums", checksums_file,
        ])
        assert result == 0

    def test_validate_with_failures_returns_one(self, nfs_tree_str, tmp_path):
        checksums_file = str(tmp_path / "checksums.json")
        with open(checksums_file, "w") as f:
            json.dump({}, f)

        result = main([
            "--nfs-root", nfs_tree_str,
            "--validate",
            "--s3-checksums", checksums_file,
        ])
        assert result == 1

    def test_validate_without_s3_checksums_fails(self, nfs_tree_str):
        result = main(["--nfs-root", nfs_tree_str, "--validate"])
        assert result == 1

    def test_resume_without_manifest_fails(self, tmp_path):
        checksums_file = str(tmp_path / "c.json")
        with open(checksums_file, "w") as f:
            json.dump({}, f)
        result = main(["--resume", "--s3-checksums", checksums_file])
        assert result == 1

    def test_resume_without_s3_checksums_fails(self, tmp_path):
        result = main(["--resume", "--manifest", str(tmp_path / "m.json")])
        assert result == 1

    def test_json_output(self, nfs_tree_str, tmp_path, capsys):
        checksums_file = str(tmp_path / "checksums.json")
        # Build matching checksums
        v = MigrationValidator(nfs_root=nfs_tree_str)
        records = v.scan_nfs()
        checksums = {(r.s3_key or r.path): r.sha256 for r in records}
        with open(checksums_file, "w") as f:
            json.dump(checksums, f)

        result = main([
            "--nfs-root", nfs_tree_str,
            "--validate",
            "--s3-checksums", checksums_file,
            "--json",
        ])
        assert result == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["verified"] == 3

    def test_output_writes_report_file(self, nfs_tree_str, tmp_path):
        checksums_file = str(tmp_path / "checksums.json")
        v = MigrationValidator(nfs_root=nfs_tree_str)
        records = v.scan_nfs()
        checksums = {(r.s3_key or r.path): r.sha256 for r in records}
        with open(checksums_file, "w") as f:
            json.dump(checksums, f)

        out = str(tmp_path / "report.txt")
        result = main([
            "--nfs-root", nfs_tree_str,
            "--validate",
            "--s3-checksums", checksums_file,
            "--output", out,
        ])
        assert result == 0
        assert os.path.exists(out)

    def test_extensions_filter_cli(self, nfs_tree_str, tmp_path):
        out = str(tmp_path / "manifest.json")
        result = main([
            "--nfs-root", nfs_tree_str,
            "--generate-manifest",
            "--output", out,
            "--extensions", ".pdf",
        ])
        assert result == 0
        with open(out) as f:
            data = json.load(f)
        assert data["file_count"] == 1
