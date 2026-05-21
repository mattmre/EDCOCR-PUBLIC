"""Output manifest builder for OCR job results.

Scans a job's output directory structure and builds a manifest of all
produced artifacts with their types, paths, sizes, and schema versions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from schemas import SCHEMA_VERSION
except ImportError:
    SCHEMA_VERSION = "1.0"

# Map EXPORT/ subdirectory names to output types
OUTPUT_DIR_MAP: dict[str, str] = {
    "PDF": "searchable_pdf",
    "TEXT": "ocr_text",
    "STRUCTURE": "structure",
    "ENTITIES": "entities",
    "NER": "ner",
    "EXTRACTION": "extraction",
    "CLASSIFICATION": "classification",
    "VALIDATION": "validation",
    "HANDWRITING": "handwriting",
    "SIGNATURE": "signature",
    "VERTICAL": "vertical",
    "RETRIEVAL": "retrieval",
}

# File extension patterns for non-EXPORT artifacts
EXTENSION_MAP: dict[str, str] = {
    ".custody.jsonl": "custody",
}

# MIME types for known output types
_MIME_TYPES: dict[str, str] = {
    "searchable_pdf": "application/pdf",
    "ocr_text": "text/plain",
    "structure": "application/json",
    "entities": "application/json",
    "ner": "application/json",
    "extraction": "application/json",
    "classification": "application/json",
    "validation": "application/json",
    "handwriting": "application/json",
    "signature": "application/json",
    "vertical": "application/json",
    "retrieval": "application/json",
    "custody": "application/jsonl",
}

# Valid output types for the retrieval endpoint
VALID_OUTPUT_TYPES = frozenset(OUTPUT_DIR_MAP.values()) | frozenset(EXTENSION_MAP.values())


@dataclass
class OutputArtifact:
    """A single output artifact produced by the pipeline."""

    output_type: str
    filename: str
    relative_path: str
    size_bytes: int
    mime_type: str = "application/octet-stream"
    schema_version: str = SCHEMA_VERSION


@dataclass
class OutputManifest:
    """Manifest of all output artifacts for a job."""

    job_id: str
    output_dir: str
    artifacts: list[OutputArtifact] = field(default_factory=list)
    schema_versions: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "job_id": self.job_id,
            "artifacts": [
                {
                    "output_type": a.output_type,
                    "filename": a.filename,
                    "relative_path": a.relative_path,
                    "size_bytes": a.size_bytes,
                    "mime_type": a.mime_type,
                    "schema_version": a.schema_version,
                }
                for a in self.artifacts
            ],
            "schema_versions": dict(self.schema_versions),
        }


def build_manifest(job_id: str, output_dir: str) -> OutputManifest:
    """Build output manifest by scanning job's output directory.

    Scans ``output_dir/EXPORT/<subdir>/`` for each known subdirectory
    in :data:`OUTPUT_DIR_MAP`, and also checks for custody files via
    :data:`EXTENSION_MAP`.

    Args:
        job_id: The job identifier.
        output_dir: Absolute path to the job's output directory.

    Returns:
        An :class:`OutputManifest` populated with discovered artifacts.
    """
    manifest = OutputManifest(job_id=job_id, output_dir=output_dir)
    export_dir = Path(output_dir) / "EXPORT"

    if not export_dir.is_dir():
        logger.debug("No EXPORT directory found at %s", export_dir)
    else:
        # Scan EXPORT subdirectories
        for dir_name, output_type in OUTPUT_DIR_MAP.items():
            subdir = export_dir / dir_name
            if not subdir.is_dir():
                continue
            for file_path in sorted(subdir.iterdir()):
                if not file_path.is_file():
                    continue
                try:
                    size = file_path.stat().st_size
                except OSError:
                    logger.warning("Could not stat file: %s", file_path)
                    continue

                relative = str(file_path.relative_to(output_dir))
                # Normalize path separators for consistency
                relative = relative.replace("\\", "/")
                mime = _MIME_TYPES.get(output_type, "application/octet-stream")

                artifact = OutputArtifact(
                    output_type=output_type,
                    filename=file_path.name,
                    relative_path=relative,
                    size_bytes=size,
                    mime_type=mime,
                )
                manifest.artifacts.append(artifact)
                manifest.schema_versions.setdefault(output_type, SCHEMA_VERSION)

    # Scan for custody files in the output root
    output_path = Path(output_dir)
    for ext_pattern, output_type in EXTENSION_MAP.items():
        for file_path in sorted(output_path.glob(f"*{ext_pattern}")):
            if not file_path.is_file():
                continue
            try:
                size = file_path.stat().st_size
            except OSError:
                logger.warning("Could not stat file: %s", file_path)
                continue

            relative = str(file_path.relative_to(output_dir))
            relative = relative.replace("\\", "/")
            mime = _MIME_TYPES.get(output_type, "application/octet-stream")

            artifact = OutputArtifact(
                output_type=output_type,
                filename=file_path.name,
                relative_path=relative,
                size_bytes=size,
                mime_type=mime,
            )
            manifest.artifacts.append(artifact)
            manifest.schema_versions.setdefault(output_type, SCHEMA_VERSION)

    return manifest
