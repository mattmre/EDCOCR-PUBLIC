"""Tests for compute_source_fingerprint_fast.

Validates that the fast stat-based fingerprint used in the scheduler
produces deterministic, collision-resistant IDs without reading file
contents.
"""

from __future__ import annotations

import hashlib
import os
import time
from unittest import mock

import pytest

# Try direct import; fall back to extraction for CI without GPU deps
try:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from ocr_gpu_async import (
        build_resume_doc_id,
        compute_source_fingerprint_fast,
    )
except ImportError:
    pytest.skip("Cannot import ocr_gpu_async", allow_module_level=True)


class TestFastFingerprintDeterminism:
    """compute_source_fingerprint_fast must be deterministic for the same file."""

    def test_same_file_gives_same_id(self, tmp_path):
        """Identical path+stat must produce identical fingerprints."""
        f = tmp_path / "document.pdf"
        f.write_bytes(b"%PDF-1.4\nSome content\n%%EOF\n")

        fp1 = compute_source_fingerprint_fast(str(f))
        fp2 = compute_source_fingerprint_fast(str(f))

        assert fp1 == fp2
        assert fp1["content_sha256"] == fp2["content_sha256"]
        assert fp1["size_bytes"] == fp2["size_bytes"]
        assert fp1["mtime_ns"] == fp2["mtime_ns"]

    def test_fingerprint_contains_required_keys(self, tmp_path):
        """Fingerprint dict must contain all three canonical keys."""
        f = tmp_path / "sample.pdf"
        f.write_bytes(b"%PDF-1.4\n%%EOF\n")

        fp = compute_source_fingerprint_fast(str(f))

        assert "content_sha256" in fp
        assert "size_bytes" in fp
        assert "mtime_ns" in fp

    def test_content_sha256_is_64_hex_chars(self, tmp_path):
        """The content_sha256 field must be a 64-character lowercase hex string."""
        f = tmp_path / "sample.pdf"
        f.write_bytes(b"test data for hashing")

        fp = compute_source_fingerprint_fast(str(f))
        sha = fp["content_sha256"]

        assert len(sha) == 64
        assert all(c in "0123456789abcdef" for c in sha)

    def test_size_bytes_matches_stat(self, tmp_path):
        """size_bytes must match os.stat() result."""
        content = b"%PDF-1.7\nSome arbitrary bytes here\n%%EOF\n"
        f = tmp_path / "sized.pdf"
        f.write_bytes(content)

        fp = compute_source_fingerprint_fast(str(f))

        assert fp["size_bytes"] == len(content)
        assert fp["size_bytes"] == os.stat(str(f)).st_size


class TestFastFingerprintCollisionResistance:
    """Different files must produce different fingerprints."""

    def test_different_paths_give_different_ids(self, tmp_path):
        """Two files at different paths must produce different fingerprints."""
        f1 = tmp_path / "doc_a.pdf"
        f2 = tmp_path / "doc_b.pdf"
        f1.write_bytes(b"%PDF-1.4\nContent A\n%%EOF\n")
        f2.write_bytes(b"%PDF-1.4\nContent B\n%%EOF\n")

        fp1 = compute_source_fingerprint_fast(str(f1))
        fp2 = compute_source_fingerprint_fast(str(f2))

        assert fp1["content_sha256"] != fp2["content_sha256"]

    def test_same_content_different_paths_differ(self, tmp_path):
        """Same content at different paths must still differ (path is part of key)."""
        f1 = tmp_path / "copy_a.pdf"
        f2 = tmp_path / "copy_b.pdf"
        content = b"%PDF-1.4\nIdentical\n%%EOF\n"
        f1.write_bytes(content)
        f2.write_bytes(content)

        fp1 = compute_source_fingerprint_fast(str(f1))
        fp2 = compute_source_fingerprint_fast(str(f2))

        assert fp1["content_sha256"] != fp2["content_sha256"]

    def test_modified_file_changes_fingerprint(self, tmp_path):
        """Rewriting a file changes both mtime and fingerprint."""
        f = tmp_path / "mutable.pdf"
        f.write_bytes(b"%PDF-1.4\nOriginal\n%%EOF\n")
        fp1 = compute_source_fingerprint_fast(str(f))

        # Ensure mtime changes (some filesystems have 1s resolution)
        time.sleep(0.05)
        f.write_bytes(b"%PDF-1.4\nModified content here\n%%EOF\n")
        fp2 = compute_source_fingerprint_fast(str(f))

        # At least one of size or mtime must differ
        assert fp1["size_bytes"] != fp2["size_bytes"] or fp1["mtime_ns"] != fp2["mtime_ns"]
        assert fp1["content_sha256"] != fp2["content_sha256"]


class TestFastFingerprintNoFileRead:
    """The fast fingerprint must never open or read the file."""

    def test_no_open_called(self, tmp_path):
        """Verify that compute_source_fingerprint_fast does not call open()."""
        f = tmp_path / "notouch.pdf"
        f.write_bytes(b"%PDF-1.4\n%%EOF\n")

        stat_result = os.stat(str(f))
        with mock.patch("ocr_gpu_async.os.stat", return_value=stat_result) as mock_stat, \
             mock.patch("builtins.open", side_effect=AssertionError("open() must not be called")):
            fp = compute_source_fingerprint_fast(str(f))

        mock_stat.assert_called_once_with(str(f))
        assert "content_sha256" in fp
        assert fp["size_bytes"] == stat_result.st_size

    def test_works_with_mock_stat_only(self):
        """A purely mocked os.stat proves no disk I/O beyond stat."""
        fake_stat = mock.Mock()
        fake_stat.st_size = 12345678
        fake_stat.st_mtime_ns = 1700000000000000000
        fake_stat.st_mtime = 1700000000.0  # fallback for getattr default eval

        fake_path = "/nonexistent/large_file.pdf"
        with mock.patch("ocr_gpu_async.os.stat", return_value=fake_stat):
            fp = compute_source_fingerprint_fast(fake_path)

        expected_key = f"{fake_path}:{fake_stat.st_mtime_ns}:{fake_stat.st_size}"
        expected_hash = hashlib.sha256(expected_key.encode("utf-8")).hexdigest()

        assert fp["content_sha256"] == expected_hash
        assert fp["size_bytes"] == 12345678
        assert fp["mtime_ns"] == 1700000000000000000


class TestFastFingerprintIntegration:
    """Integration with build_resume_doc_id."""

    def test_build_resume_doc_id_accepts_fast_fingerprint(self, tmp_path):
        """Fast fingerprint must be compatible with build_resume_doc_id."""
        f = tmp_path / "integration.pdf"
        f.write_bytes(b"%PDF-1.4\nIntegration test\n%%EOF\n")

        fp = compute_source_fingerprint_fast(str(f))
        doc_id = build_resume_doc_id(str(f), fp)

        assert isinstance(doc_id, str)
        assert len(doc_id) == 16
        assert all(c in "0123456789abcdef" for c in doc_id)

    def test_resume_doc_id_is_deterministic(self, tmp_path):
        """Same file must produce same doc_id across multiple calls."""
        f = tmp_path / "stable.pdf"
        f.write_bytes(b"%PDF-1.4\nStable\n%%EOF\n")

        fp1 = compute_source_fingerprint_fast(str(f))
        fp2 = compute_source_fingerprint_fast(str(f))
        id1 = build_resume_doc_id(str(f), fp1)
        id2 = build_resume_doc_id(str(f), fp2)

        assert id1 == id2

    def test_different_files_produce_different_doc_ids(self, tmp_path):
        """Different files must produce different doc_ids."""
        f1 = tmp_path / "alpha.pdf"
        f2 = tmp_path / "beta.pdf"
        f1.write_bytes(b"%PDF alpha")
        f2.write_bytes(b"%PDF beta")

        fp1 = compute_source_fingerprint_fast(str(f1))
        fp2 = compute_source_fingerprint_fast(str(f2))
        id1 = build_resume_doc_id(str(f1), fp1)
        id2 = build_resume_doc_id(str(f2), fp2)

        assert id1 != id2
