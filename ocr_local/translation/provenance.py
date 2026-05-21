"""Translation model provenance validation -- Plan B Wave M2 PR B19.

Implements the E-B-008 / RED-07 requirement that every bundled MT model
ships with a verifiable supply-chain attestation chain:

* **SLSA v1.0 provenance statement** -- a URI pointing at the build
  attestation (``slsa_provenance_uri``).
* **In-toto attestation digest** -- SHA-256 of the in-toto envelope
  describing what was built and how (``intoto_attestation_sha256``).
* **CycloneDX SBOM digest** -- SHA-256 of the CycloneDX software bill
  of materials for the model bundle (``sbom_sha256``).

The validator works in two modes:

* ``enforce=False`` (default at config level) -- used by the legacy
  stub flows (``model_dir=""``) so existing tests and air-gapped
  bring-up continue to work.
* ``enforce=True`` -- production: any engine binding without all three
  fields raises :class:`ProvenanceMissingError` and the engine registry
  refuses to register that engine instance.

Stdlib-only on purpose so this module sits underneath the optional
``ctranslate2`` dependency tree and is importable on the SDK CI lane.
The companion CLI :mod:`scripts.generate_model_provenance` writes the
canonical ``provenance.json`` next to a model directory; this module
reads it back via :func:`load_provenance_from_dir`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "PROVENANCE_FILE_NAME",
    "ModelProvenance",
    "ProvenanceMissingError",
    "ProvenanceCorruptError",
    "compute_file_sha256",
    "load_provenance_from_dir",
    "validate_provenance",
    "validate_engine_provenance",
]


PROVENANCE_FILE_NAME = "provenance.json"

# Required keys in the provenance dict that gate engine registration
# when ``enforce=True``.  These names match the
# ``TRANSLATION_APPLIED`` custody-event payload in
# ``docs/planning/2026-04-24-translation-swarm/02-plan-B-translation-pipeline.md``
# section 7 (per E-B-008 / RED-07).
REQUIRED_SLSA_FIELDS: tuple[str, ...] = (
    "slsa_provenance_uri",
    "intoto_attestation_sha256",
    "sbom_sha256",
)

# Required ancillary fields -- always present, both with and without
# enforcement, because they document the binding even in legacy mode.
REQUIRED_BASE_FIELDS: tuple[str, ...] = (
    "weights_sha256",
    "license",
    "runtime_version",
)

_SHA256_HEX_LEN = 64


class ProvenanceMissingError(RuntimeError):
    """Raised when a model lacks one or more SLSA/in-toto/SBOM fields."""


class ProvenanceCorruptError(RuntimeError):
    """Raised when a ``provenance.json`` file is malformed or truncated."""


@dataclasses.dataclass(frozen=True)
class ModelProvenance:
    """Validated provenance record for a translation model bundle.

    All hex digest fields are lower-case 64-char SHA-256 strings.  The
    ``slsa_provenance_uri`` is a verbatim URI (typically ``https://``
    or ``oci://``); validation only checks presence and non-emptiness.
    """

    slsa_provenance_uri: str
    intoto_attestation_sha256: str
    sbom_sha256: str
    weights_sha256: str
    license: str
    runtime_version: str

    def to_dict(self) -> dict[str, str]:
        return dataclasses.asdict(self)


def compute_file_sha256(path: str | Path) -> str:
    """Stream a file through SHA-256 and return the lower-case hex digest."""

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_hex64(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != _SHA256_HEX_LEN:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _coerce_str_field(field: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProvenanceMissingError(
            f"provenance field {field!r} must be a non-empty string"
        )
    return value


def load_provenance_from_dir(model_dir: str | Path) -> dict[str, Any]:
    """Read ``provenance.json`` from ``model_dir`` and return the parsed dict.

    Raises :class:`FileNotFoundError` if the file is missing and
    :class:`ProvenanceCorruptError` if the JSON cannot be parsed or is
    not a top-level object.
    """

    path = Path(model_dir) / PROVENANCE_FILE_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"provenance.json missing at {path}; run "
            "scripts/generate_model_provenance.py for the model bundle"
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProvenanceCorruptError(
            f"provenance.json at {path} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ProvenanceCorruptError(
            f"provenance.json at {path} must be a JSON object, got "
            f"{type(data).__name__}"
        )
    return data


def validate_provenance(
    provenance: Mapping[str, Any],
    *,
    enforce: bool = True,
) -> ModelProvenance:
    """Validate a provenance dict and return a :class:`ModelProvenance`.

    When ``enforce=True`` the SLSA/in-toto/SBOM triple is required.
    When ``enforce=False`` missing SLSA fields are tolerated -- they
    are filled with the sentinel string ``"unverified"`` so downstream
    custody events still record a value rather than skipping the
    field.  The ancillary base fields (weights, license, runtime) are
    always required.

    Hex-digest fields are normalised to lower case and checked to be
    well-formed 64-char SHA-256 hex strings.
    """

    if not isinstance(provenance, Mapping):
        raise ProvenanceMissingError(
            f"provenance must be a mapping, got {type(provenance).__name__}"
        )

    # Always-required ancillary fields.
    weights = _coerce_str_field("weights_sha256", provenance.get("weights_sha256"))
    license_ = _coerce_str_field("license", provenance.get("license"))
    runtime_version = _coerce_str_field(
        "runtime_version", provenance.get("runtime_version")
    )

    if weights != "not_loaded" and not _is_hex64(weights.lower()):
        raise ProvenanceMissingError(
            "weights_sha256 must be 64-char lower-case hex SHA-256 (or "
            "the sentinel 'not_loaded' for stub engines)"
        )

    # SLSA / in-toto / SBOM triple.
    slsa_uri = provenance.get("slsa_provenance_uri")
    intoto_sha = provenance.get("intoto_attestation_sha256")
    sbom_sha = provenance.get("sbom_sha256")

    missing = [
        field
        for field, value in (
            ("slsa_provenance_uri", slsa_uri),
            ("intoto_attestation_sha256", intoto_sha),
            ("sbom_sha256", sbom_sha),
        )
        if not isinstance(value, str) or not value.strip()
    ]

    if missing:
        if enforce:
            raise ProvenanceMissingError(
                "model provenance missing required SLSA/in-toto/SBOM "
                f"fields: {', '.join(missing)}; "
                "run scripts/generate_model_provenance.py or set "
                "translation_enforce_provenance=False to bypass during "
                "bring-up"
            )
        # Non-enforced mode: stamp sentinel values for any missing
        # field so downstream custody events still get a string.
        if not isinstance(slsa_uri, str) or not slsa_uri.strip():
            slsa_uri = "unverified"
        if not isinstance(intoto_sha, str) or not intoto_sha.strip():
            intoto_sha = "unverified"
        if not isinstance(sbom_sha, str) or not sbom_sha.strip():
            sbom_sha = "unverified"

    # Hex-digest validation -- only when present and not the sentinel.
    for field, value in (
        ("intoto_attestation_sha256", intoto_sha),
        ("sbom_sha256", sbom_sha),
    ):
        if value != "unverified" and not _is_hex64(value.lower()):
            raise ProvenanceMissingError(
                f"{field} must be 64-char lower-case hex SHA-256 "
                f"(or 'unverified' when enforce=False)"
            )

    return ModelProvenance(
        slsa_provenance_uri=slsa_uri,
        intoto_attestation_sha256=intoto_sha.lower()
        if intoto_sha != "unverified"
        else intoto_sha,
        sbom_sha256=sbom_sha.lower() if sbom_sha != "unverified" else sbom_sha,
        weights_sha256=weights.lower() if weights != "not_loaded" else weights,
        license=license_,
        runtime_version=runtime_version,
    )


def validate_engine_provenance(
    engine: Any,
    *,
    enforce: bool | None = None,
) -> ModelProvenance:
    """Validate an engine instance's :meth:`model_provenance` output.

    Reads the active :class:`pipeline_config.PipelineConfig` to decide
    the default ``enforce`` value when the caller does not pass one.
    Falls back to ``enforce=False`` if the pipeline config is not yet
    initialised so legacy / SDK lanes that import the engines without
    bootstrapping the pipeline config keep working.
    """

    if enforce is None:
        try:
            from pipeline_config import get_config  # type: ignore[import-not-found]

            cfg = get_config()
            enforce = bool(getattr(cfg, "translation_enforce_provenance", False))
        except (ImportError, AttributeError, RuntimeError):
            enforce = False

    raw = engine.model_provenance()
    if not isinstance(raw, Mapping):
        raise ProvenanceMissingError(
            f"engine {engine.__class__.__name__}.model_provenance() must "
            f"return a mapping, got {type(raw).__name__}"
        )
    return validate_provenance(raw, enforce=enforce)
