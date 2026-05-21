"""Shared helpers for versioned EDC bundle contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CONTRACT_SCHEMA_NAMES = frozenset(
    {
        "document-bundle-v1",
        "translation-bundle-v1",
    }
)

_SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas"


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    """Return deterministic UTF-8 JSON bytes for hashing bundle payloads."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    """Return SHA-256 over deterministic JSON bytes."""

    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_hex(data: bytes) -> str:
    """Return lowercase SHA-256 hex for raw bytes."""

    return hashlib.sha256(data).hexdigest()


def load_contract_schema(schema_name: str) -> dict[str, Any]:
    """Load a known bundle schema by logical name."""

    if schema_name not in CONTRACT_SCHEMA_NAMES:
        raise ValueError(
            f"Unknown EDC contract schema: {schema_name!r}. "
            f"Valid: {sorted(CONTRACT_SCHEMA_NAMES)}"
        )
    schema_path = _SCHEMA_DIR / f"{schema_name}.schema.json"
    with schema_path.open(encoding="utf-8") as fh:
        return json.load(fh)


def validate_contract_payload(
    payload: dict[str, Any],
    schema_name: str,
) -> dict[str, Any]:
    """Validate *payload* against a bundle contract and return it unchanged."""

    try:
        import jsonschema  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "jsonschema is required to validate EDC bundle contracts"
        ) from exc

    schema = load_contract_schema(schema_name)
    jsonschema.Draft7Validator.check_schema(schema)
    jsonschema.validate(payload, schema)
    return payload
