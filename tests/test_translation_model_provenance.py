"""Tests for ``ocr_local.translation.provenance`` (Plan B Wave M2 PR B19).

These tests are stdlib-only on purpose so they run on the SDK CI lane
without ``ctranslate2`` or any other translation runtime installed.
Coverage spans:

* :func:`validate_provenance` enforce / non-enforce branches
* :func:`load_provenance_from_dir` happy-path + missing / malformed
* :func:`compute_file_sha256` round-trip vs hashlib
* :func:`validate_engine_provenance` against a fake engine
* End-to-end exercise of ``scripts/generate_model_provenance.py``

The script-level tests build a tiny model directory, an in-toto
attestation file, and a CycloneDX SBOM, then invoke the script's
``main()`` directly so no subprocess is needed.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest

from ocr_local.translation.provenance import (
    PROVENANCE_FILE_NAME,
    REQUIRED_BASE_FIELDS,
    REQUIRED_SLSA_FIELDS,
    ModelProvenance,
    ProvenanceCorruptError,
    ProvenanceMissingError,
    compute_file_sha256,
    load_provenance_from_dir,
    validate_engine_provenance,
    validate_provenance,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _good_payload(**overrides):
    base = {
        "slsa_provenance_uri": "https://models.example/opus-mt/slsa-v1.intoto.jsonl",
        "intoto_attestation_sha256": "a" * 64,
        "sbom_sha256": "b" * 64,
        "weights_sha256": "c" * 64,
        "license": "CC-BY-4.0",
        "runtime_version": "4.6.1",
    }
    base.update(overrides)
    return base


class _FakeEngine:
    """Engine stand-in that returns whatever provenance dict we hand it."""

    __class__ = type("FakeEngine", (), {})  # for nicer repr in errors

    def __init__(self, prov):
        self._prov = prov

    def model_provenance(self):
        return self._prov


# ----------------------------------------------------------------------
# REQUIRED_*_FIELDS contract
# ----------------------------------------------------------------------


def test_required_field_constants_match_planning_doc():
    """Constants here are the canonical names used by the B19 PR row."""

    assert REQUIRED_SLSA_FIELDS == (
        "slsa_provenance_uri",
        "intoto_attestation_sha256",
        "sbom_sha256",
    )
    assert REQUIRED_BASE_FIELDS == (
        "weights_sha256",
        "license",
        "runtime_version",
    )


# ----------------------------------------------------------------------
# compute_file_sha256
# ----------------------------------------------------------------------


def test_compute_file_sha256_matches_hashlib(tmp_path):
    blob = b"hello world\n"
    path = tmp_path / "weights.bin"
    path.write_bytes(blob)
    expected = hashlib.sha256(blob).hexdigest()
    assert compute_file_sha256(path) == expected


def test_compute_file_sha256_streaming_handles_large_input(tmp_path):
    # Two MiB > the 1 MiB streaming chunk so the inner loop fires.
    blob = b"x" * (2 * 1024 * 1024 + 7)
    path = tmp_path / "big.bin"
    path.write_bytes(blob)
    assert compute_file_sha256(path) == hashlib.sha256(blob).hexdigest()


# ----------------------------------------------------------------------
# validate_provenance enforce=True (default)
# ----------------------------------------------------------------------


def test_validate_provenance_accepts_complete_record():
    rec = validate_provenance(_good_payload())
    assert isinstance(rec, ModelProvenance)
    assert rec.intoto_attestation_sha256 == "a" * 64
    assert rec.sbom_sha256 == "b" * 64
    assert rec.weights_sha256 == "c" * 64
    assert rec.license == "CC-BY-4.0"
    assert rec.runtime_version == "4.6.1"


@pytest.mark.parametrize("missing", REQUIRED_SLSA_FIELDS)
def test_validate_provenance_rejects_missing_slsa_field(missing):
    payload = _good_payload()
    payload.pop(missing)
    with pytest.raises(ProvenanceMissingError) as excinfo:
        validate_provenance(payload, enforce=True)
    assert missing in str(excinfo.value)


@pytest.mark.parametrize("missing", REQUIRED_SLSA_FIELDS)
def test_validate_provenance_rejects_blank_slsa_field(missing):
    payload = _good_payload()
    payload[missing] = "   "
    with pytest.raises(ProvenanceMissingError):
        validate_provenance(payload, enforce=True)


def test_validate_provenance_rejects_non_hex_intoto_digest():
    payload = _good_payload(intoto_attestation_sha256="not-a-hash")
    with pytest.raises(ProvenanceMissingError) as excinfo:
        validate_provenance(payload, enforce=True)
    assert "intoto_attestation_sha256" in str(excinfo.value)


def test_validate_provenance_rejects_short_sbom_digest():
    payload = _good_payload(sbom_sha256="b" * 63)
    with pytest.raises(ProvenanceMissingError):
        validate_provenance(payload, enforce=True)


def test_validate_provenance_rejects_blank_license():
    payload = _good_payload(license="")
    with pytest.raises(ProvenanceMissingError):
        validate_provenance(payload, enforce=True)


def test_validate_provenance_rejects_non_string_runtime_version():
    payload = _good_payload(runtime_version=42)
    with pytest.raises(ProvenanceMissingError):
        validate_provenance(payload, enforce=True)


def test_validate_provenance_rejects_non_mapping_input():
    with pytest.raises(ProvenanceMissingError):
        validate_provenance(["not", "a", "mapping"])  # type: ignore[arg-type]


def test_validate_provenance_lowercases_hex_digests():
    payload = _good_payload(
        intoto_attestation_sha256="A" * 64,
        sbom_sha256="B" * 64,
        weights_sha256="C" * 64,
    )
    rec = validate_provenance(payload, enforce=True)
    assert rec.intoto_attestation_sha256 == "a" * 64
    assert rec.sbom_sha256 == "b" * 64
    assert rec.weights_sha256 == "c" * 64


def test_validate_provenance_accepts_not_loaded_weights_sentinel():
    """The OPUS-MT stub flow sets weights_sha256='not_loaded' until binding."""

    payload = _good_payload(weights_sha256="not_loaded")
    rec = validate_provenance(payload, enforce=True)
    assert rec.weights_sha256 == "not_loaded"


# ----------------------------------------------------------------------
# validate_provenance enforce=False (legacy / bring-up)
# ----------------------------------------------------------------------


def test_validate_provenance_enforce_false_fills_unverified():
    payload = _good_payload()
    for field in REQUIRED_SLSA_FIELDS:
        payload.pop(field)
    rec = validate_provenance(payload, enforce=False)
    assert rec.slsa_provenance_uri == "unverified"
    assert rec.intoto_attestation_sha256 == "unverified"
    assert rec.sbom_sha256 == "unverified"
    # Base fields still required even in enforce=False mode.
    assert rec.license == "CC-BY-4.0"


def test_validate_provenance_enforce_false_still_requires_base_fields():
    payload = _good_payload()
    payload.pop("license")
    with pytest.raises(ProvenanceMissingError):
        validate_provenance(payload, enforce=False)


# ----------------------------------------------------------------------
# load_provenance_from_dir
# ----------------------------------------------------------------------


def test_load_provenance_from_dir_round_trip(tmp_path):
    payload = _good_payload()
    (tmp_path / PROVENANCE_FILE_NAME).write_text(json.dumps(payload))
    loaded = load_provenance_from_dir(tmp_path)
    assert loaded == payload


def test_load_provenance_from_dir_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_provenance_from_dir(tmp_path)


def test_load_provenance_from_dir_malformed_json(tmp_path):
    (tmp_path / PROVENANCE_FILE_NAME).write_text("{not valid json")
    with pytest.raises(ProvenanceCorruptError):
        load_provenance_from_dir(tmp_path)


def test_load_provenance_from_dir_top_level_must_be_object(tmp_path):
    (tmp_path / PROVENANCE_FILE_NAME).write_text("[1,2,3]")
    with pytest.raises(ProvenanceCorruptError):
        load_provenance_from_dir(tmp_path)


# ----------------------------------------------------------------------
# validate_engine_provenance
# ----------------------------------------------------------------------


def test_validate_engine_provenance_accepts_good_engine():
    engine = _FakeEngine(_good_payload())
    rec = validate_engine_provenance(engine, enforce=True)
    assert rec.license == "CC-BY-4.0"


def test_validate_engine_provenance_rejects_bad_engine_when_enforced():
    payload = _good_payload()
    payload.pop("slsa_provenance_uri")
    engine = _FakeEngine(payload)
    with pytest.raises(ProvenanceMissingError):
        validate_engine_provenance(engine, enforce=True)


def test_validate_engine_provenance_bypass_when_enforce_false():
    payload = _good_payload()
    payload.pop("slsa_provenance_uri")
    engine = _FakeEngine(payload)
    rec = validate_engine_provenance(engine, enforce=False)
    assert rec.slsa_provenance_uri == "unverified"


def test_validate_engine_provenance_rejects_non_mapping_provenance():
    engine = _FakeEngine("not-a-dict")
    with pytest.raises(ProvenanceMissingError):
        validate_engine_provenance(engine, enforce=False)


def test_validate_engine_provenance_default_enforce_follows_pipeline_config():
    """When ``enforce=None`` the helper consults pipeline_config.

    We exercise the fallback path that fires when pipeline_config is
    not initialised -- the helper must default to ``enforce=False`` so
    the stub flows used by the SDK lane keep working.
    """

    payload = _good_payload()
    payload.pop("slsa_provenance_uri")
    engine = _FakeEngine(payload)
    # No call to set_pipeline_config -- get_config() raises RuntimeError
    # which the helper catches and treats as enforce=False.
    rec = validate_engine_provenance(engine)
    assert rec.slsa_provenance_uri == "unverified"


# ----------------------------------------------------------------------
# scripts/generate_model_provenance.py end-to-end
# ----------------------------------------------------------------------


def _import_generator():
    """Import the script as a module so we can call ``main()`` directly."""

    repo_root = Path(__file__).resolve().parent.parent
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    return importlib.import_module("generate_model_provenance")


def _make_model_bundle(tmp_path: Path) -> tuple[Path, Path, Path]:
    model_dir = tmp_path / "opus-mt-en-fr"
    model_dir.mkdir()
    (model_dir / "model.bin").write_bytes(b"weights-blob")
    (model_dir / "vocab.txt").write_text("hello\nworld\n")
    intoto = tmp_path / "opus-mt-en-fr.intoto.jsonl"
    intoto.write_text(
        json.dumps(
            {
                "_type": "https://in-toto.io/Statement/v1",
                "subject": [{"name": "model.bin", "digest": {"sha256": "..."}}],
            }
        )
        + "\n"
    )
    sbom = tmp_path / "opus-mt-en-fr.cdx.json"
    sbom.write_text(
        json.dumps(
            {
                "bomFormat": "CycloneDX",
                "specVersion": "1.5",
                "components": [],
            }
        )
    )
    return model_dir, intoto, sbom


def test_generate_writes_validated_provenance_json(tmp_path):
    gen = _import_generator()
    model_dir, intoto, sbom = _make_model_bundle(tmp_path)
    rc = gen.main(
        [
            "--model-dir",
            str(model_dir),
            "--slsa-uri",
            "https://models.example/opus-mt-en-fr/slsa.intoto.jsonl",
            "--intoto-attestation",
            str(intoto),
            "--sbom",
            str(sbom),
            "--license",
            "CC-BY-4.0",
            "--runtime-version",
            "4.6.1",
        ]
    )
    assert rc == 0
    out = model_dir / PROVENANCE_FILE_NAME
    assert out.exists()
    payload = json.loads(out.read_text())

    # Validator must accept what the generator wrote.
    rec = validate_provenance(payload, enforce=True)
    assert rec.license == "CC-BY-4.0"
    assert rec.intoto_attestation_sha256 == compute_file_sha256(intoto)
    assert rec.sbom_sha256 == compute_file_sha256(sbom)


def test_generate_aggregate_weights_digest_is_deterministic(tmp_path):
    gen = _import_generator()
    model_dir, intoto, sbom = _make_model_bundle(tmp_path)
    args1 = [
        "--model-dir",
        str(model_dir),
        "--slsa-uri",
        "https://models.example/x/slsa.intoto.jsonl",
        "--intoto-attestation",
        str(intoto),
        "--sbom",
        str(sbom),
        "--license",
        "CC-BY-4.0",
        "--runtime-version",
        "4.6.1",
    ]
    gen.main(args1)
    first = json.loads((model_dir / PROVENANCE_FILE_NAME).read_text())
    # Re-run with --overwrite -- weights digest must be byte-identical.
    gen.main(args1 + ["--overwrite"])
    second = json.loads((model_dir / PROVENANCE_FILE_NAME).read_text())
    assert first["weights_sha256"] == second["weights_sha256"]


def test_generate_excludes_provenance_json_from_weights_digest(tmp_path):
    """Re-running without --overwrite would otherwise include the previous
    provenance.json in the next aggregate, drifting the digest."""

    gen = _import_generator()
    model_dir, intoto, sbom = _make_model_bundle(tmp_path)
    base_args = [
        "--model-dir",
        str(model_dir),
        "--slsa-uri",
        "https://models.example/x/slsa.intoto.jsonl",
        "--intoto-attestation",
        str(intoto),
        "--sbom",
        str(sbom),
        "--license",
        "CC-BY-4.0",
        "--runtime-version",
        "4.6.1",
    ]
    gen.main(base_args)
    digest_first = json.loads(
        (model_dir / PROVENANCE_FILE_NAME).read_text()
    )["weights_sha256"]
    # Modify provenance.json -- subsequent runs must ignore it.
    (model_dir / PROVENANCE_FILE_NAME).write_text("{}")
    gen.main(base_args + ["--overwrite"])
    digest_second = json.loads(
        (model_dir / PROVENANCE_FILE_NAME).read_text()
    )["weights_sha256"]
    assert digest_first == digest_second


def test_generate_refuses_overwrite_without_flag(tmp_path):
    gen = _import_generator()
    model_dir, intoto, sbom = _make_model_bundle(tmp_path)
    (model_dir / PROVENANCE_FILE_NAME).write_text("{}")
    with pytest.raises(SystemExit) as exc:
        gen.main(
            [
                "--model-dir",
                str(model_dir),
                "--slsa-uri",
                "https://models.example/x/slsa.intoto.jsonl",
                "--intoto-attestation",
                str(intoto),
                "--sbom",
                str(sbom),
                "--license",
                "CC-BY-4.0",
                "--runtime-version",
                "4.6.1",
            ]
        )
    assert exc.value.code == gen.EXIT_ARG_ERROR


def test_generate_rejects_missing_attestation(tmp_path):
    gen = _import_generator()
    model_dir, _intoto, sbom = _make_model_bundle(tmp_path)
    with pytest.raises(SystemExit) as exc:
        gen.main(
            [
                "--model-dir",
                str(model_dir),
                "--slsa-uri",
                "https://models.example/x/slsa.intoto.jsonl",
                "--intoto-attestation",
                str(tmp_path / "missing.intoto.jsonl"),
                "--sbom",
                str(sbom),
                "--license",
                "CC-BY-4.0",
                "--runtime-version",
                "4.6.1",
            ]
        )
    assert exc.value.code == gen.EXIT_ARG_ERROR


def test_generate_rejects_missing_model_dir(tmp_path):
    gen = _import_generator()
    intoto = tmp_path / "i.jsonl"
    intoto.write_text("{}")
    sbom = tmp_path / "s.json"
    sbom.write_text("{}")
    with pytest.raises(SystemExit) as exc:
        gen.main(
            [
                "--model-dir",
                str(tmp_path / "does_not_exist"),
                "--slsa-uri",
                "https://models.example/x/slsa.intoto.jsonl",
                "--intoto-attestation",
                str(intoto),
                "--sbom",
                str(sbom),
                "--license",
                "CC-BY-4.0",
                "--runtime-version",
                "4.6.1",
            ]
        )
    assert exc.value.code == gen.EXIT_ARG_ERROR


# ----------------------------------------------------------------------
# pipeline_config wiring
# ----------------------------------------------------------------------


def test_pipeline_config_default_translation_enforce_provenance_is_false():
    pc = importlib.import_module("pipeline_config")
    cfg = pc.PipelineConfig()
    assert cfg.translation_enforce_provenance is False


def test_pipeline_config_translation_enforce_provenance_round_trip():
    pc = importlib.import_module("pipeline_config")
    cfg = pc.PipelineConfig(translation_enforce_provenance=True)
    assert cfg.translation_enforce_provenance is True


def test_pipeline_config_env_loader_reads_provenance_flag(monkeypatch):
    pc = importlib.import_module("pipeline_config")
    monkeypatch.setenv("OCR_TRANSLATION_ENFORCE_PROVENANCE", "true")
    cfg = pc.create_pipeline_config()
    assert cfg.translation_enforce_provenance is True
