"""Sidecar writer for translation output -- Plan B Wave M1.

Writes the ``.translation.json`` sidecar (and a human-readable Markdown
summary) for a finalised :class:`DocumentTranslation`.  The JSON layout
is constrained by ``schemas/translation.schema.json`` and is validated
prior to write when ``jsonschema`` is installed.

The most important invariant enforced here is that
``DocumentTranslation.certified`` is ``False`` in raw sidecar output --
certification is an explicit downstream attestation performed by the
review queue (piv_cac / oidc_mfa / hardware_token).  Any attempt to
write a sidecar with ``certified=True`` raises :class:`SchemaValidationError`.
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ocr_local.translation.models import DocumentTranslation

__all__ = [
    "SchemaValidationError",
    "write_translation_json",
    "write_translation_md",
]

_SCHEMA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "schemas", "translation.schema.json"
)
_SCHEMA_VERSION = "1.0"


class SchemaValidationError(ValueError):
    """Raised when a ``DocumentTranslation`` violates the sidecar schema."""


def _load_schema() -> dict:
    """Load the translation JSON Schema from disk."""
    with open(os.path.normpath(_SCHEMA_PATH), encoding="utf-8") as f:
        return json.load(f)


def _validate_doc(doc: "DocumentTranslation") -> None:
    """Validate ``doc`` against the translation schema.

    Always enforces the ``certified is False`` invariant.  When
    ``jsonschema`` is installed, additionally validates the full
    serialised payload against ``translation.schema.json``.
    """
    # Always enforce certified=False invariant -- this MUST run even
    # when jsonschema is unavailable.
    if doc.certified:
        raise SchemaValidationError(
            "certified must be False in sidecar output (review queue "
            "promotes via auth)"
        )

    try:
        import jsonschema  # type: ignore[import-not-found]
    except ImportError:
        return  # jsonschema is optional

    schema = _load_schema()
    payload = dataclasses.asdict(doc)
    try:
        jsonschema.validate(payload, schema)
    except jsonschema.ValidationError as exc:  # pragma: no cover - thin wrap
        raise SchemaValidationError(str(exc)) from exc


def _ensure_subfolder(output_dir: str, subfolder: str) -> str:
    """Resolve and create the destination directory under EXPORT/TRANSLATION."""
    base = os.path.join(output_dir, "EXPORT", "TRANSLATION")
    target = os.path.join(base, subfolder) if subfolder else base
    os.makedirs(target, exist_ok=True)
    return target


def write_translation_json(
    doc: "DocumentTranslation",
    output_dir: str,
    subfolder: str = "",
) -> str:
    """Serialise ``doc`` to ``EXPORT/TRANSLATION/<subfolder>/<id>.<tgt>.translation.json``.

    Returns the absolute path of the written file.  Raises
    :class:`SchemaValidationError` when ``doc.certified`` is True or
    (when ``jsonschema`` is installed) the payload otherwise violates
    the schema.
    """
    _validate_doc(doc)

    target_dir = _ensure_subfolder(output_dir, subfolder)
    filename = f"{doc.document_id}.{doc.target_language}.translation.json"
    json_path = os.path.join(target_dir, filename)

    payload = dataclasses.asdict(doc)
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, indent=2))

    return os.path.abspath(json_path)


def write_translation_md(
    doc: "DocumentTranslation",
    output_dir: str,
    subfolder: str = "",
) -> str:
    """Write a human-readable Markdown summary alongside the JSON sidecar.

    Returns the absolute path of the written file.  Does not run schema
    validation -- ``write_translation_json`` is the gating writer.
    """
    target_dir = _ensure_subfolder(output_dir, subfolder)
    filename = f"{doc.document_id}.{doc.target_language}.translation.md"
    md_path = os.path.join(target_dir, filename)

    span_count = sum(len(p.spans) for p in doc.pages)
    engine_id = doc.engine.get("id", "") if isinstance(doc.engine, dict) else ""
    quality = doc.quality if isinstance(doc.quality, dict) else {}
    mean_score = quality.get("mean_score")
    below_threshold = quality.get("below_threshold_count", 0)
    quality_class = quality.get("quality_class", "")

    lines = [
        f"# Translation: {doc.document_id}",
        "",
        f"- source_file: {doc.source_file}",
        f"- source_language: {doc.source_language}",
        f"- target_language: {doc.target_language}",
        f"- engine: {engine_id}",
        f"- certified: {str(bool(doc.certified)).lower()}",
        f"- pages: {len(doc.pages)}",
        f"- spans: {span_count}",
        "",
        "## Quality",
        "",
        f"- mean_score: {mean_score}",
        f"- below_threshold_count: {below_threshold}",
        f"- quality_class: {quality_class}",
        "",
    ]

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return os.path.abspath(md_path)
