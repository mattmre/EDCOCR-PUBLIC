"""Tests for api.output_manifest — output manifest builder."""

from __future__ import annotations

from api.output_manifest import (
    EXTENSION_MAP,
    OUTPUT_DIR_MAP,
    VALID_OUTPUT_TYPES,
    OutputArtifact,
    OutputManifest,
    build_manifest,
)

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
            "PDF", "TEXT", "STRUCTURE", "ENTITIES", "NER",
            "EXTRACTION", "CLASSIFICATION", "VALIDATION",
            "HANDWRITING", "SIGNATURE", "VERTICAL", "RETRIEVAL",
        }
        assert set(OUTPUT_DIR_MAP.keys()) == expected_dirs

    def test_covers_expected_output_types(self):
        expected_types = {
            "searchable_pdf", "ocr_text", "structure", "entities", "ner",
            "extraction", "classification", "validation",
            "handwriting", "signature", "vertical", "retrieval",
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
        (retrieval_dir / "document.retrieval.json").write_text('{"schema_version":"1.0"}')

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
            ext = ".pdf" if dir_name == "PDF" else ".json" if dir_name != "TEXT" else ".txt"
            (subdir / f"doc{ext}").write_bytes(b"content")

        m = build_manifest("job_aaa111bbb222", str(tmp_path))
        assert len(m.artifacts) == 4
        types = {a.output_type for a in m.artifacts}
        assert types == {"searchable_pdf", "ocr_text", "ner", "validation"}

    def test_scan_custody_file(self, tmp_path):
        """Finds custody JSONL files in the output root."""
        (tmp_path / "EXPORT").mkdir()
        (tmp_path / "job.custody.jsonl").write_text('{"event":"created"}\n')

        m = build_manifest("job_aaa111bbb222", str(tmp_path))
        assert len(m.artifacts) == 1
        assert m.artifacts[0].output_type == "custody"
        assert m.artifacts[0].mime_type == "application/jsonl"

    def test_scan_custody_file_without_export_dir(self, tmp_path):
        """Custody artifacts remain retrievable even when no EXPORT tree exists."""
        (tmp_path / "job.custody.jsonl").write_text('{"event":"created"}\n')

        m = build_manifest("job_aaa111bbb222", str(tmp_path))
        assert len(m.artifacts) == 1
        assert m.artifacts[0].output_type == "custody"
        assert m.artifacts[0].relative_path == "job.custody.jsonl"

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
