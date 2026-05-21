"""Tests for production translation evidence validators."""

from __future__ import annotations

import json
import hashlib

from scripts import validate_plugin_runtime_attestation as plugin_attest
from scripts import validate_translation_cluster_evidence as cluster
from scripts import validate_translation_model_approval as model_approval
from scripts import validate_translation_tsa_assurance as tsa_assurance
from ocr_local.translation.tsa import _der_integer, _der_sequence


def test_model_approval_missing_artifact_fails(tmp_path):
    assert model_approval.main(["--approval", str(tmp_path / "missing.json")]) == 1


def test_model_approval_requires_operator_approval_fields(tmp_path):
    approval = tmp_path / "approval.json"
    approval.write_text(json.dumps({"schema_version": "translation-model-approval-v1"}), encoding="utf-8")

    assert model_approval.main(["--approval", str(approval)]) == 1


def test_model_approval_accepts_complete_fixture(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.bin").write_bytes(b"model weights")
    digest = "a" * 64
    (model_dir / "provenance.json").write_text(
        json.dumps(
            {
                "weights_sha256": digest,
                "license": "Apache-2.0",
                "runtime_version": "ct2-fixture",
                "slsa_provenance_uri": "https://example.invalid/slsa.intoto.jsonl",
                "intoto_attestation_sha256": digest,
                "sbom_sha256": digest,
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "MODEL_SHA256").write_text("b" * 64, encoding="utf-8")
    quality = tmp_path / "quality.json"
    quality.write_text(
        json.dumps({"exact_match_rate": 1.0, "segments": 200}),
        encoding="utf-8",
    )
    provenance_hash = hashlib.sha256((model_dir / "provenance.json").read_bytes()).hexdigest()
    approval = tmp_path / "approval.json"
    approval.write_text(
        json.dumps(
            {
                "schema_version": "translation-model-approval-v1",
                "approval_status": "approved",
                "approved_by": "operator-fixture",
                "approval_ticket": "TICKET-FIXTURE",
                "model_dir": "model",
                "model_id": "fixture-model",
                "source_url": "https://example.invalid/model",
                "source_revision": "fixture-revision",
                "license": "Apache-2.0",
                "conversion_command": "fixture conversion command",
                "provenance_json_sha256": provenance_hash,
                "model_bundle_sha256": "b" * 64,
                "quality_report_path": "quality.json",
                "min_exact_match_rate": 0.99,
                "min_segments": 200,
            }
        ),
        encoding="utf-8",
    )

    assert model_approval.main(["--approval", str(approval)]) == 0


def test_tsa_assurance_rejects_fake_tsr(tmp_path):
    tsr = tmp_path / "fake.tsr"
    tsr.write_text("not der", encoding="utf-8")
    for name in ("chain.pem", "trust.pem", "verify.txt"):
        (tmp_path / name).write_text("operator evidence", encoding="utf-8")
    evidence = tmp_path / "tsa.json"
    evidence.write_text(
        json.dumps(
            {
                "schema_version": "translation-tsa-assurance-v1",
                "approval_status": "approved",
                "tsa_url": "http://timestamp.digicert.com",
                "tsa_policy_oid": "2.16.840.1.114412.7.1",
                "approved_by": "operator",
                "approval_ticket": "TICKET-1",
                "tsr_path": "fake.tsr",
                "certificate_chain_path": "chain.pem",
                "trust_store_path": "trust.pem",
                "verification_log_path": "verify.txt",
            }
        ),
        encoding="utf-8",
    )

    assert tsa_assurance.main(["--evidence", str(evidence)]) == 1


def test_tsa_assurance_accepts_parseable_dev_tsr_fixture(tmp_path):
    tsr = tmp_path / "dev.tsr"
    tsr.write_bytes(_der_sequence(_der_sequence(_der_integer(0))))
    for name in ("chain.pem", "trust.pem", "verify.txt"):
        (tmp_path / name).write_text("operator fixture evidence", encoding="utf-8")
    evidence = tmp_path / "tsa.json"
    evidence.write_text(
        json.dumps(
            {
                "schema_version": "translation-tsa-assurance-v1",
                "approval_status": "approved",
                "tsa_url": "http://timestamp.digicert.com",
                "tsa_policy_oid": "2.16.840.1.114412.7.1",
                "approved_by": "operator-fixture",
                "approval_ticket": "TICKET-FIXTURE",
                "tsr_path": "dev.tsr",
                "certificate_chain_path": "chain.pem",
                "trust_store_path": "trust.pem",
                "verification_log_path": "verify.txt",
            }
        ),
        encoding="utf-8",
    )

    assert tsa_assurance.main(["--evidence", str(evidence)]) == 0


def test_cluster_evidence_requires_two_production_clusters(tmp_path):
    evidence = tmp_path / "cluster.json"
    evidence.write_text(
        json.dumps(
            {
                "schema_version": "translation-cluster-evidence-v1",
                "environment": "production",
                "clusters": ["primary"],
            }
        ),
        encoding="utf-8",
    )

    assert cluster.main(["--evidence", str(evidence)]) == 1


def test_plugin_runtime_attestation_accepts_complete_evidence(tmp_path):
    evidence = tmp_path / "plugin.json"
    evidence.write_text(
        json.dumps(
            {
                "schema_version": "plugin-runtime-attestation-v1",
                "environment": "production",
                "runner_id": "runner",
                "runner_version": "1.0.0",
                "workload_identity": "svc/plugin",
                "timestamp": "2026-05-12T00:00:00Z",
                "image_digest": "sha256:" + "a" * 64,
                "attestation_signature_sha256": "b" * 64,
                "out_of_process": True,
                "network_disabled": True,
                "read_only_rootfs": True,
                "non_root_user": True,
                "cpu_limit_enforced": True,
                "memory_limit_enforced": True,
                "wall_time_limit_enforced": True,
                "no_new_privileges": True,
                "capabilities_dropped": True,
                "seccomp_profile": "docker-default",
                "adversarial_escape_tests_passed": True,
            }
        ),
        encoding="utf-8",
    )

    assert plugin_attest.main(["--attestation", str(evidence)]) == 0


def test_plugin_runtime_attestation_fails_open_network(tmp_path):
    evidence = tmp_path / "plugin.json"
    evidence.write_text(
        json.dumps(
            {
                "schema_version": "plugin-runtime-attestation-v1",
                "environment": "production",
                "runner_id": "runner",
                "runner_version": "1.0.0",
                "workload_identity": "svc/plugin",
                "timestamp": "2026-05-12T00:00:00Z",
                "image_digest": "sha256:" + "a" * 64,
                "attestation_signature_sha256": "b" * 64,
                "out_of_process": True,
                "network_disabled": False,
                "read_only_rootfs": True,
                "non_root_user": True,
                "cpu_limit_enforced": True,
                "memory_limit_enforced": True,
                "wall_time_limit_enforced": True,
                "no_new_privileges": True,
                "capabilities_dropped": True,
                "seccomp_profile": "docker-default",
                "adversarial_escape_tests_passed": True,
            }
        ),
        encoding="utf-8",
    )

    assert plugin_attest.main(["--attestation", str(evidence)]) == 1


def test_cluster_evidence_accepts_local_dev_only_with_explicit_flag(tmp_path):
    for name, text in {
        "helm.txt": "STATUS: dry-run\nMANIFEST:\nkind: Deployment\n",
        "failover.txt": "local failover rehearsal",
        "residency.txt": "local residency rehearsal",
        "mtls.txt": "local mtls identity rehearsal",
    }.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    evidence = tmp_path / "cluster-local.json"
    evidence.write_text(
        json.dumps(
            {
                "schema_version": "translation-cluster-evidence-v1",
                "environment": "local-dev",
                "clusters": ["ocr-local-sandbox"],
                "helm_install_or_upgrade_passed": True,
                "outbox_failover_drill_passed": False,
                "residency_routing_drill_passed": False,
                "mtls_identity_drill_passed": False,
                "federation_health_passed": False,
                "helm_log_path": "helm.txt",
                "failover_log_path": "failover.txt",
                "residency_log_path": "residency.txt",
                "mtls_log_path": "mtls.txt",
            }
        ),
        encoding="utf-8",
    )

    assert cluster.main(["--evidence", str(evidence)]) == 1
    assert cluster.main(["--evidence", str(evidence), "--allow-local-dev"]) == 0


def test_plugin_runtime_attestation_accepts_local_dev_only_with_explicit_flag(tmp_path):
    evidence = tmp_path / "plugin-local.json"
    evidence.write_text(
        json.dumps(
            {
                "schema_version": "plugin-runtime-attestation-v1",
                "environment": "local-dev",
                "runner_id": "local-docker-dev-runner",
                "runner_version": "docker-cli",
                "workload_identity": "local-dev-plugin-proof",
                "timestamp": "2026-05-12T00:00:00Z",
                "image_digest": "sha256:" + "a" * 64,
                "attestation_signature_sha256": "b" * 64,
                "out_of_process": True,
                "network_disabled": True,
                "read_only_rootfs": True,
                "non_root_user": True,
                "cpu_limit_enforced": True,
                "memory_limit_enforced": True,
                "wall_time_limit_enforced": True,
                "no_new_privileges": True,
                "capabilities_dropped": True,
                "seccomp_profile": "docker-default",
                "adversarial_escape_tests_passed": True,
            }
        ),
        encoding="utf-8",
    )

    assert plugin_attest.main(["--attestation", str(evidence)]) == 1
    assert plugin_attest.main(["--attestation", str(evidence), "--allow-local-dev"]) == 0
