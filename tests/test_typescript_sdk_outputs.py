"""Tests for TypeScript SDK output and schema declarations.

These are content-based tests that verify the TypeScript SDK file contains
the expected interface definitions and method signatures. No TypeScript
compiler or runtime is required.
"""

from pathlib import Path

import pytest

TS_SDK_PATH = Path(__file__).resolve().parent.parent / "sdk" / "typescript" / "ocr_client.ts"


@pytest.fixture(scope="module")
def ts_content() -> str:
    """Read the TypeScript SDK source file."""
    return TS_SDK_PATH.read_text(encoding="utf-8")


# ------------------------------------------------------------------
# Interface declarations
# ------------------------------------------------------------------


class TestOutputArtifactInterface:
    """Verify OutputArtifact interface exists with required fields."""

    def test_interface_declared(self, ts_content):
        assert "export interface OutputArtifact" in ts_content

    def test_has_output_type_field(self, ts_content):
        assert "output_type: string" in ts_content

    def test_has_filename_field(self, ts_content):
        # OutputArtifact has filename; check it appears after the interface
        idx = ts_content.index("export interface OutputArtifact")
        block = ts_content[idx:idx + 500]
        assert "filename: string" in block

    def test_has_relative_path_field(self, ts_content):
        idx = ts_content.index("export interface OutputArtifact")
        block = ts_content[idx:idx + 500]
        assert "relative_path: string" in block

    def test_has_size_bytes_field(self, ts_content):
        idx = ts_content.index("export interface OutputArtifact")
        block = ts_content[idx:idx + 500]
        assert "size_bytes: number" in block

    def test_has_schema_version_field(self, ts_content):
        idx = ts_content.index("export interface OutputArtifact")
        block = ts_content[idx:idx + 500]
        assert "schema_version: string" in block

    def test_has_optional_mime_type(self, ts_content):
        idx = ts_content.index("export interface OutputArtifact")
        block = ts_content[idx:idx + 500]
        assert "mime_type?: string" in block


class TestOutputManifestInterface:
    """Verify OutputManifest interface exists with required fields."""

    def test_interface_declared(self, ts_content):
        assert "export interface OutputManifest" in ts_content

    def test_has_job_id_field(self, ts_content):
        idx = ts_content.index("export interface OutputManifest")
        block = ts_content[idx:idx + 400]
        assert "job_id: string" in block

    def test_has_artifacts_field(self, ts_content):
        idx = ts_content.index("export interface OutputManifest")
        block = ts_content[idx:idx + 400]
        assert "artifacts: OutputArtifact[]" in block

    def test_has_schema_versions_field(self, ts_content):
        idx = ts_content.index("export interface OutputManifest")
        block = ts_content[idx:idx + 400]
        assert "schema_versions: Record<string, string>" in block


class TestSchemaListItemInterface:
    """Verify SchemaListItem interface exists with required fields."""

    def test_interface_declared(self, ts_content):
        assert "export interface SchemaListItem" in ts_content

    def test_has_output_type_field(self, ts_content):
        idx = ts_content.index("export interface SchemaListItem")
        block = ts_content[idx:idx + 300]
        assert "output_type: string" in block

    def test_has_schema_version_field(self, ts_content):
        idx = ts_content.index("export interface SchemaListItem")
        block = ts_content[idx:idx + 300]
        assert "schema_version: string" in block


# ------------------------------------------------------------------
# Method signatures
# ------------------------------------------------------------------


class TestGetOutputsMethod:
    """Verify getOutputs method exists in OcrClient."""

    def test_method_declared(self, ts_content):
        assert "async getOutputs(jobId: string): Promise<OutputManifest>" in ts_content

    def test_calls_correct_endpoint(self, ts_content):
        assert "/api/v1/jobs/${jobId}/outputs" in ts_content


class TestGetOutputMethod:
    """Verify getOutput method exists in OcrClient."""

    def test_method_declared(self, ts_content):
        assert "async getOutput(jobId: string, outputType: string): Promise<ArrayBuffer>" in ts_content

    def test_calls_correct_endpoint(self, ts_content):
        assert "/api/v1/jobs/${jobId}/outputs/${outputType}" in ts_content


class TestGetOutputJsonMethod:
    """Verify getOutputJson method exists in OcrClient."""

    def test_method_declared(self, ts_content):
        assert "async getOutputJson(jobId: string, outputType: string): Promise<Record<string, unknown>>" in ts_content


class TestListSchemasMethod:
    """Verify listSchemas method exists in OcrClient."""

    def test_method_declared(self, ts_content):
        assert "async listSchemas(): Promise<SchemaListItem[]>" in ts_content

    def test_calls_correct_endpoint(self, ts_content):
        assert "/api/v1/schemas" in ts_content


class TestGetSchemaMethod:
    """Verify getSchema method exists in OcrClient."""

    def test_method_declared(self, ts_content):
        assert "async getSchema(outputType: string): Promise<Record<string, unknown>>" in ts_content

    def test_calls_correct_endpoint(self, ts_content):
        assert "/api/v1/schemas/${outputType}" in ts_content
