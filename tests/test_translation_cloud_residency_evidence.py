"""Tests for cloud translation residency evidence validator."""

from __future__ import annotations

import json

from scripts import validate_translation_cloud_residency_evidence as validator


def test_missing_evidence_fails(tmp_path):
    assert validator.main(["--evidence", str(tmp_path / "missing.json")]) == 1


def test_ai_studio_production_fails(tmp_path):
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "deployment_env": "production",
                "gemini_provider": "ai_studio",
                "ai_studio_enabled": True,
                "region": "us-central1",
                "retention_policy": "zero_retention",
                "training_disabled": True,
            }
        ),
        encoding="utf-8",
    )

    assert validator.main(["--evidence", str(evidence)]) == 1


def test_phi_legal_production_requires_baa_path(tmp_path):
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "deployment_env": "production",
                "gemini_provider": "vertex",
                "ai_studio_enabled": False,
                "region": "us-central1",
                "retention_policy": "zero_retention",
                "training_disabled": True,
                "phi_or_legal_production": True,
            }
        ),
        encoding="utf-8",
    )

    assert validator.main(["--evidence", str(evidence)]) == 1


def test_valid_vertex_evidence_passes(tmp_path):
    baa = tmp_path / "baa.txt"
    baa.write_text("operator-supplied BAA evidence placeholder", encoding="utf-8")
    compliance = tmp_path / "zero-retention.txt"
    compliance.write_text("operator-supplied zero-retention evidence", encoding="utf-8")
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "deployment_env": "production",
                "gemini_provider": "vertex",
                "ai_studio_enabled": False,
                "region": "us-central1",
                "retention_policy": "zero_retention",
                "training_disabled": True,
                "phi_or_legal_production": True,
                "baa_evidence_path": "baa.txt",
                "compliance_evidence_path": "zero-retention.txt",
                "routing_policy_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )

    assert validator.main(["--evidence", str(evidence)]) == 0


def test_compliance_evidence_path_must_exist(tmp_path):
    baa = tmp_path / "baa.txt"
    baa.write_text("operator-supplied BAA evidence placeholder", encoding="utf-8")
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "deployment_env": "production",
                "gemini_provider": "vertex",
                "ai_studio_enabled": False,
                "region": "us-central1",
                "retention_policy": "zero_retention",
                "training_disabled": True,
                "phi_or_legal_production": True,
                "baa_evidence_path": "baa.txt",
                "compliance_evidence_path": "missing.txt",
                "routing_policy_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )

    assert validator.main(["--evidence", str(evidence)]) == 1


def test_routing_policy_sha_is_required(tmp_path):
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "deployment_env": "production",
                "gemini_provider": "vertex",
                "ai_studio_enabled": False,
                "region": "us-central1",
                "retention_policy": "zero_retention",
                "training_disabled": True,
                "phi_or_legal_production": False,
            }
        ),
        encoding="utf-8",
    )

    assert validator.main(["--evidence", str(evidence)]) == 1


def test_unexpected_fields_fail_schema(tmp_path):
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "deployment_env": "production",
                "gemini_provider": "vertex",
                "ai_studio_enabled": False,
                "region": "us-central1",
                "retention_policy": "zero_retention",
                "training_disabled": True,
                "phi_or_legal_production": False,
                "routing_policy_sha256": "a" * 64,
                "unexpected": "not allowed",
            }
        ),
        encoding="utf-8",
    )

    assert validator.main(["--evidence", str(evidence)]) == 1
