"""Tests for api.output_manifest — output manifest builder."""

from __future__ import annotations

import os

import pytest

from api.output_manifest import (
    EXTENSION_MAP,
    OUTPUT_DIR_MAP,
    VALID_OUTPUT_TYPES,
    OutputArtifact,
    OutputManifest,
    build_manifest,
)
from schemas import ARTIFACT_OUTPUT_TYPES

# ---------------------------------------------------------------------------
# OutputArtifact dataclass
# ---------------------------------------------------------------------------


class TestOutputArtifact:
    def test_create_artifact(self):
        a = OutputArtifact(
            output_type="ocr_text",
            filename="doc.txt",
            relative_path="EXPORT/TEXT/doc.txt",
            size_bytes=1234,
        )
        assert a.output_type == "ocr_text"
        assert a.filename == "doc.txt"
        assert a.relative_path == "EXPORT/TEXT/doc.txt"
        assert a.size_bytes == 1234
        assert a.mime_type == "application/octet-stream"

    def test_artifact_with_mime_type(self):
        a = OutputArtifact(
            output_type="searchable_pdf",
            filename="doc.pdf",
            relative_path="EXPORT/PDF/doc.pdf",
            size_bytes=999,
            mime_type="application/pdf",
        )
        assert a.mime_type == "application/pdf"

    def test_artifact_schema_version_set(self):
        a = OutputArtifact(
            output_type="ner",
            filename="doc.ner.json",
            relative_path="EXPORT/NER/doc.ner.json",
            size_bytes=10,
        )
        assert a.schema_version  # Should have a default value


# ---------------------------------------------------------------------------
# OutputManifest dataclass
# ---------------------------------------------------------------------------


class TestOutputManifest:
    def test_empty_manifest(self):
        m = OutputManifest(job_id="job_abc123def456", output_dir="/tmp/out")
        assert m.job_id == "job_abc123def456"
        assert m.artifacts == []
        assert m.schema_versions == {}

    def test_manifest_to_dict(self):
        m = OutputManifest(
            job_id="job_abc123def456",
            output_dir="/tmp/out",
            artifacts=[
                OutputArtifact(
                    output_type="ocr_text",
                    filename="doc.txt",
                    relative_path="EXPORT/TEXT/doc.txt",
                    size_bytes=100,
                    mime_type="text/plain",
                ),
            ],
            schema_versions={"ocr_text": "1.0"},
        )
        d = m.to_dict()
        assert d["job_id"] == "job_abc123def456"
        assert len(d["artifacts"]) == 1
        assert d["artifacts"][0]["output_type"] == "ocr_text"
        assert d["artifacts"][0]["size_bytes"] == 100
        assert d["artifacts"][0]["mime_type"] == "text/plain"
        assert d["schema_versions"] == {"ocr_text": "1.0"}

    def test_manifest_to_dict_empty(self):
        m = OutputManifest(job_id="job_000000000000", output_dir="/tmp/out")
        d = m.to_dict()
        assert d["artifacts"] == []
        assert d["schema_versions"] == {}


# ---------------------------------------------------------------------------
# OUTPUT_DIR_MAP coverage
# ---------------------------------------------------------------------------


class TestOutputDirMap:
    def test_covers_expected_directories(self):
        expected_dirs = {
            "PDF",
            "TEXT",
            "STRUCTURE",
            "ENTITIES",
            "NER",
            "EXTRACTION",
            "CLASSIFICATION",
            "VALIDATION",
            "HANDWRITING",
            "SIGNATURE",
            "VERTICAL",
            "RETRIEVAL",
            "FORM",
            "AI_TEXT",
            "LANGUAGE",
            "SYMBOLOGY",
            "CUSTODY",
        }
        assert set(OUTPUT_DIR_MAP.keys()) == expected_dirs

    def test_covers_expected_output_types(self):
        expected_types = {
            "searchable_pdf",
            "ocr_text",
            "structure",
            "entities",
            "ner",
            "extraction",
            "classification",
            "validation",
            "handwriting",
            "signature",
            "vertical",
            "retrieval",
            "form",
            "ai_text",
            "language",
            "symbology",
            "custody",
        }
        assert set(OUTPUT_DIR_MAP.values()) == expected_types

    def test_extension_map_has_custody(self):
        assert ".custody.jsonl" in EXTENSION_MAP
        assert EXTENSION_MAP[".custody.jsonl"] == "custody"

    def test_valid_output_types_superset(self):
        """VALID_OUTPUT_TYPES includes all dir-mapped and extension-mapped types."""
        for ot in OUTPUT_DIR_MAP.values():
            assert ot in VALID_OUTPUT_TYPES
        for ot in EXTENSION_MAP.values():
            assert ot in VALID_OUTPUT_TYPES

    def test_downloadable_types_match_schema_artifact_contract(self):
        assert VALID_OUTPUT_TYPES == frozenset(ARTIFACT_OUTPUT_TYPES)


# ---------------------------------------------------------------------------
# build_manifest()
# ---------------------------------------------------------------------------


class TestBuildManifest:
    def test_empty_dir(self, tmp_path):
        """No EXPORT directory => empty manifest."""
        m = build_manifest("job_aaa111bbb222", str(tmp_path))
        assert m.job_id == "job_aaa111bbb222"
        assert m.artifacts == []

    def test_empty_export_dir(self, tmp_path):
        """EXPORT directory exists but empty."""
        (tmp_path / "EXPORT").mkdir()
        m = build_manifest("job_aaa111bbb222", str(tmp_path))
        assert m.artifacts == []

    def test_scan_pdf_dir(self, tmp_path):
        """Finds PDF files in EXPORT/PDF/."""
        pdf_dir = tmp_path / "EXPORT" / "PDF"
        pdf_dir.mkdir(parents=True)
        (pdf_dir / "document.pdf").write_bytes(b"%PDF-1.0 test content")

        m = build_manifest("job_aaa111bbb222", str(tmp_path))
        assert len(m.artifacts) == 1
        a = m.artifacts[0]
        assert a.output_type == "searchable_pdf"
        assert a.filename == "document.pdf"
        assert a.relative_path == "EXPORT/PDF/document.pdf"
        assert a.size_bytes > 0
        assert a.mime_type == "application/pdf"
        assert "searchable_pdf" in m.schema_versions

    def test_scan_text_dir(self, tmp_path):
        """Finds text files in EXPORT/TEXT/."""
        text_dir = tmp_path / "EXPORT" / "TEXT"
        text_dir.mkdir(parents=True)
        (text_dir / "document.txt").write_text("Hello world")

        m = build_manifest("job_aaa111bbb222", str(tmp_path))
        assert len(m.artifacts) == 1
        assert m.artifacts[0].output_type == "ocr_text"
        assert m.artifacts[0].mime_type == "text/plain"

    def test_scan_retrieval_dir(self, tmp_path):
        """Finds retrieval JSON files in EXPORT/RETRIEVAL/."""
        retrieval_dir = tmp_path / "EXPORT" / "RETRIEVAL"
        retrieval_dir.mkdir(parents=True)
        (retrieval_dir / "document.retrieval.json").write_text(
            '{"schema_version":"1.0"}'
        )

        m = build_manifest("job_aaa111bbb222", str(tmp_path))
        assert len(m.artifacts) == 1
        a = m.artifacts[0]
        assert a.output_type == "retrieval"
        assert a.filename == "document.retrieval.json"
        assert a.relative_path == "EXPORT/RETRIEVAL/document.retrieval.json"
        assert a.size_bytes > 0
        assert a.mime_type == "application/json"
        assert "retrieval" in m.schema_versions

    def test_scan_multiple_dirs(self, tmp_path):
        """Finds artifacts across multiple EXPORT subdirectories."""
        for dir_name in ["PDF", "TEXT", "NER", "VALIDATION"]:
            subdir = tmp_path / "EXPORT" / dir_name
            subdir.mkdir(parents=True)
            ext = (
                ".pdf"
                if dir_name == "PDF"
                else ".json"
                if dir_name != "TEXT"
                else ".txt"
            )
            (subdir / f"doc{ext}").write_bytes(b"content")

        m = build_manifest("job_aaa111bbb222", str(tmp_path))
        assert len(m.artifacts) == 4
        types = {a.output_type for a in m.artifacts}
        assert types == {"searchable_pdf", "ocr_text", "ner", "validation"}

    def test_scans_preserved_nested_source_folders(self, tmp_path):
        language_dir = tmp_path / "EXPORT" / "LANGUAGE" / "matter" / "exhibit-a"
        language_dir.mkdir(parents=True)
        artifact = language_dir / "document.language.json"
        artifact.write_text('{"schema_version":"1.0"}')

        manifest = build_manifest("job_aaa111bbb222", str(tmp_path))

        assert len(manifest.artifacts) == 1
        assert manifest.artifacts[0].output_type == "language"
        assert manifest.artifacts[0].relative_path == (
            "EXPORT/LANGUAGE/matter/exhibit-a/document.language.json"
        )

    def test_scan_custody_file(self, tmp_path):
        """Finds custody JSONL files in the output root."""
        (tmp_path / "EXPORT").mkdir()
        (tmp_path / "job.custody.jsonl").write_text('{"event":"created"}\n')

        m = build_manifest("job_aaa111bbb222", str(tmp_path))
        assert len(m.artifacts) == 1
        assert m.artifacts[0].output_type == "custody"
        assert m.artifacts[0].mime_type == "application/jsonl"

    def test_skips_symlinked_artifact_metadata(self, tmp_path):
        export = tmp_path / "EXPORT" / "TEXT"
        export.mkdir(parents=True)
        foreign = tmp_path / "foreign.txt"
        foreign.write_text("not this job", encoding="utf-8")
        try:
            (export / "linked.txt").symlink_to(foreign)
        except OSError:
            return

        manifest = build_manifest("job_aaa111bbb222", str(tmp_path))
        assert manifest.artifacts == []

    def test_scan_custody_file_without_export_dir(self, tmp_path):
        """Custody artifacts remain retrievable even when no EXPORT tree exists."""
        (tmp_path / "job.custody.jsonl").write_text('{"event":"created"}\n')

        m = build_manifest("job_aaa111bbb222", str(tmp_path))
        assert len(m.artifacts) == 1
        assert m.artifacts[0].output_type == "custody"
        assert m.artifacts[0].relative_path == "job.custody.jsonl"

    def test_skips_root_fifo_named_like_custody_without_blocking(self, tmp_path):
        """A FIFO at the job root must not hang manifest construction."""
        if not hasattr(os, "mkfifo"):
            pytest.skip("mkfifo is unavailable")
        (tmp_path / "EXPORT" / "TEXT").mkdir(parents=True)
        (tmp_path / "EXPORT" / "TEXT" / "doc.txt").write_text("ok", encoding="utf-8")
        os.mkfifo(tmp_path / "hang.custody.jsonl")

        manifest = build_manifest("job_aaa111bbb222", str(tmp_path))
        assert [item.output_type for item in manifest.artifacts] == ["ocr_text"]

    def test_skips_export_custody_fifo_without_blocking(self, tmp_path):
        """A FIFO under EXPORT/CUSTODY must be type-checked after a nonblocking open."""
        if not hasattr(os, "mkfifo"):
            pytest.skip("mkfifo is unavailable")
        custody_dir = tmp_path / "EXPORT" / "CUSTODY"
        custody_dir.mkdir(parents=True)
        os.mkfifo(custody_dir / "hang.custody.jsonl")
        (custody_dir / "real.custody.jsonl").write_text("{}\n", encoding="utf-8")

        manifest = build_manifest("job_aaa111bbb222", str(tmp_path))
        assert [item.filename for item in manifest.artifacts] == ["real.custody.jsonl"]
        assert manifest.artifacts[0].nlink == 1

    def test_ignores_subdirectories_in_export(self, tmp_path):
        """Subdirectories inside EXPORT/PDF/ are not treated as artifacts."""
        pdf_dir = tmp_path / "EXPORT" / "PDF"
        pdf_dir.mkdir(parents=True)
        (pdf_dir / "nested_dir").mkdir()
        (pdf_dir / "doc.pdf").write_bytes(b"content")

        m = build_manifest("job_aaa111bbb222", str(tmp_path))
        assert len(m.artifacts) == 1
        assert m.artifacts[0].filename == "doc.pdf"

    def test_ignores_unknown_export_subdirs(self, tmp_path):
        """Unknown EXPORT subdirectories are ignored."""
        unknown_dir = tmp_path / "EXPORT" / "UNKNOWN"
        unknown_dir.mkdir(parents=True)
        (unknown_dir / "file.bin").write_bytes(b"data")

        m = build_manifest("job_aaa111bbb222", str(tmp_path))
        assert m.artifacts == []

    def test_multiple_files_in_one_dir(self, tmp_path):
        """Multiple files in one EXPORT subdir are all captured."""
        text_dir = tmp_path / "EXPORT" / "TEXT"
        text_dir.mkdir(parents=True)
        (text_dir / "page1.txt").write_text("Page 1")
        (text_dir / "page2.txt").write_text("Page 2")

        m = build_manifest("job_aaa111bbb222", str(tmp_path))
        assert len(m.artifacts) == 2
        names = sorted(a.filename for a in m.artifacts)
        assert names == ["page1.txt", "page2.txt"]

    def test_schema_versions_populated(self, tmp_path):
        """Schema versions dict is populated for discovered types."""
        pdf_dir = tmp_path / "EXPORT" / "PDF"
        pdf_dir.mkdir(parents=True)
        (pdf_dir / "doc.pdf").write_bytes(b"data")

        ner_dir = tmp_path / "EXPORT" / "NER"
        ner_dir.mkdir(parents=True)
        (ner_dir / "doc.ner.json").write_bytes(b"{}")

        m = build_manifest("job_aaa111bbb222", str(tmp_path))
        assert "searchable_pdf" in m.schema_versions
        assert "ner" in m.schema_versions

    def test_relative_path_uses_forward_slashes(self, tmp_path):
        """Relative paths always use forward slashes regardless of OS."""
        text_dir = tmp_path / "EXPORT" / "TEXT"
        text_dir.mkdir(parents=True)
        (text_dir / "doc.txt").write_text("test")

        m = build_manifest("job_aaa111bbb222", str(tmp_path))
        assert "\\" not in m.artifacts[0].relative_path
        assert "/" in m.artifacts[0].relative_path

    def test_scan_has_a_hard_entry_bound(self, tmp_path, monkeypatch):
        import api.output_manifest as manifest_module

        text_dir = tmp_path / "EXPORT" / "TEXT"
        text_dir.mkdir(parents=True)
        (text_dir / "one.txt").write_text("one", encoding="utf-8")
        (text_dir / "two.txt").write_text("two", encoding="utf-8")
        monkeypatch.setattr(manifest_module, "_MAX_MANIFEST_SCAN_ENTRIES", 1)

        with pytest.raises(ValueError, match="scan exceeds"):
            build_manifest("job_aaa111bbb222", str(tmp_path))

    def test_scan_bound_exception_closes_nested_descriptors(
        self, tmp_path, monkeypatch
    ):
        import api.output_manifest as manifest_module

        text_dir = tmp_path / "EXPORT" / "TEXT" / "one" / "two"
        text_dir.mkdir(parents=True)
        (text_dir / "artifact.txt").write_text("one", encoding="utf-8")
        monkeypatch.setattr(manifest_module, "_MAX_MANIFEST_SCAN_ENTRIES", 2)
        fd_dir = "/proc/self/fd"
        if not os.path.isdir(fd_dir):
            pytest.skip("fd accounting requires procfs")
        before = len(os.listdir(fd_dir))

        with pytest.raises(ValueError, match="scan exceeds"):
            build_manifest("job_aaa111bbb222", str(tmp_path))

        assert len(os.listdir(fd_dir)) == before

    def test_scan_has_a_hard_nesting_depth_bound(self, tmp_path, monkeypatch):
        import api.output_manifest as manifest_module

        nested = tmp_path / "EXPORT" / "TEXT" / "one" / "two"
        nested.mkdir(parents=True)
        (nested / "artifact.txt").write_text("one", encoding="utf-8")
        monkeypatch.setattr(manifest_module, "_MAX_MANIFEST_SCAN_DEPTH", 1)

        with pytest.raises(ValueError, match="scan exceeds configured depth"):
            build_manifest("job_aaa111bbb222", str(tmp_path))

    def test_scan_does_not_retain_sibling_directory_descriptors(
        self, tmp_path, monkeypatch
    ):
        import api.output_manifest as manifest_module

        text_dir = tmp_path / "EXPORT" / "TEXT"
        for index in range(32):
            child = text_dir / f"child-{index:02d}"
            child.mkdir(parents=True)
            (child / "artifact.txt").write_text("one", encoding="utf-8")

        real_open = os.open
        real_close = os.close
        opened: set[int] = set()
        max_opened = 0

        def tracking_open(*args, **kwargs):
            nonlocal max_opened
            fd = real_open(*args, **kwargs)
            opened.add(fd)
            max_opened = max(max_opened, len(opened))
            return fd

        def tracking_close(fd):
            opened.discard(fd)
            return real_close(fd)

        monkeypatch.setattr(manifest_module.os, "open", tracking_open)
        monkeypatch.setattr(manifest_module.os, "close", tracking_close)

        manifest = build_manifest("job_aaa111bbb222", str(tmp_path))

        assert len(manifest.artifacts) == 32
        assert max_opened <= 4
        assert opened == set()

    def test_symlink_job_root_is_not_walked(self, tmp_path):
        victim = tmp_path / "victim"
        (victim / "EXPORT" / "PDF").mkdir(parents=True)
        (victim / "EXPORT" / "PDF" / "secret.pdf").write_bytes(b"%PDF-1.0")
        lexical = tmp_path / "job_a"
        lexical.symlink_to(victim)
        manifest = build_manifest("job_aaa111bbb222", str(lexical))
        assert manifest.artifacts == []
