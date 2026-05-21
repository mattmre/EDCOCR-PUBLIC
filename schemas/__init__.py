"""Output contract schema definitions for EDCOCR.

Provides versioned JSON Schema definitions for all pipeline output formats.
Use load_schema(output_type) to get a schema dict, or validate(data, output_type)
to check conformance.
"""

import json
from pathlib import Path

__all__ = [
    "SCHEMA_DIR",
    "OUTPUT_TYPES",
    "load_schema",
    "validate",
    "get_schema_version",
]

SCHEMA_DIR = Path(__file__).parent
SCHEMA_VERSION = "1.0"

OUTPUT_TYPES = (
    "ocr_text",
    "searchable_pdf",
    "structure",
    "entities",
    "ner",
    "extraction",
    "classification",
    "validation",
    "handwriting",
    "signature",
    "vertical",
    "custody",
    "retrieval",
    "output_manifest",
)


def load_schema(output_type: str) -> dict:
    """Load JSON Schema for the given output type."""
    if output_type not in OUTPUT_TYPES:
        raise ValueError(
            f"Unknown output type: {output_type}. Valid: {OUTPUT_TYPES}"
        )
    path = SCHEMA_DIR / f"{output_type}.schema.json"
    if not path.exists():
        raise FileNotFoundError(f"Schema not found: {path}")
    with open(path) as f:
        return json.load(f)


def validate(data: dict, output_type: str) -> list[str]:
    """Perform smoke-test level validation of *data* against the named schema.

    This is a **lightweight, built-in validator** that checks:

    * Presence of ``required`` fields.
    * Top-level type correctness (string, number, integer, array, object,
      boolean).

    It does **not** perform full JSON Schema validation (no ``$ref``
    resolution, no ``additionalProperties`` enforcement, no nested
    ``items``/``patternProperties`` checks, etc.).  For comprehensive
    validation, use the ``jsonschema`` package::

        import jsonschema
        schema = load_schema("entities")
        jsonschema.validate(instance=data, schema=schema)

    Returns:
        A list of human-readable error strings.  An empty list means the
        data passed all smoke-test checks.
    """
    errors: list[str] = []

    schema = load_schema(output_type)

    # Check required fields
    for req in schema.get("required", []):
        if req not in data:
            errors.append(f"Missing required field: {req}")
    # Check top-level types
    props = schema.get("properties", {})
    for key, val in data.items():
        if key in props:
            expected_type = props[key].get("type")
            if expected_type == "string" and not isinstance(val, str):
                errors.append(
                    f"Field '{key}' should be string, got {type(val).__name__}"
                )
            elif expected_type == "number" and not isinstance(val, (int, float)):
                errors.append(
                    f"Field '{key}' should be number, got {type(val).__name__}"
                )
            elif expected_type == "integer" and not isinstance(val, int):
                errors.append(
                    f"Field '{key}' should be integer, got {type(val).__name__}"
                )
            elif expected_type == "array" and not isinstance(val, list):
                errors.append(
                    f"Field '{key}' should be array, got {type(val).__name__}"
                )
            elif expected_type == "object" and not isinstance(val, dict):
                errors.append(
                    f"Field '{key}' should be object, got {type(val).__name__}"
                )
            elif expected_type == "boolean" and not isinstance(val, bool):
                errors.append(
                    f"Field '{key}' should be boolean, got {type(val).__name__}"
                )
    return errors


def get_schema_version() -> str:
    """Return the current schema contract version."""
    return SCHEMA_VERSION
