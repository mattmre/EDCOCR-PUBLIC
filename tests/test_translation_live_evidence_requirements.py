"""Tests for translation-swarm live evidence prerequisite checker."""

from __future__ import annotations

from scripts import check_translation_live_evidence_requirements as checker


def test_default_mode_reports_missing_without_failing(monkeypatch):
    for req in checker.REQUIREMENTS.values():
        monkeypatch.delenv(req.env_var, raising=False)

    assert checker.main([]) == 0


def test_required_missing_live_tsa_fails(monkeypatch):
    monkeypatch.delenv("TRANSLATION_TSA_URL", raising=False)
    monkeypatch.delenv("TRANSLATION_TSA_TSR_PATH", raising=False)

    assert checker.main(["--require", "live-tsa"]) == 1


def test_required_live_tsa_passes_when_present(monkeypatch):
    monkeypatch.setenv("TRANSLATION_TSA_URL", "https://tsa.example/rfc3161")

    assert checker.main(["--require", "live-tsa"]) == 0


def test_required_live_tsa_passes_with_tsr_artifact(monkeypatch, tmp_path):
    monkeypatch.delenv("TRANSLATION_TSA_URL", raising=False)
    tsr = tmp_path / "dev.tsr"
    tsr.write_bytes(b"tsr")

    assert checker.main(["--require", "live-tsa", "--tsa-tsr", str(tsr)]) == 0


def test_required_path_must_exist(monkeypatch, tmp_path):
    missing = tmp_path / "missing"
    monkeypatch.setenv("TRANSLATION_CT2_MODEL_DIR", str(missing))

    assert checker.main(["--require", "ct2-model"]) == 1

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    monkeypatch.setenv("TRANSLATION_CT2_MODEL_DIR", str(model_dir))
    assert checker.main(["--require", "ct2-model"]) == 0


def test_kubeconfig_passes_with_cluster_evidence_artifact(monkeypatch, tmp_path):
    monkeypatch.delenv("KUBECONFIG", raising=False)
    evidence = tmp_path / "kind-nodes.txt"
    evidence.write_text("node ready", encoding="utf-8")

    assert checker.main(["--require", "kubeconfig", "--cluster-evidence", str(evidence)]) == 0


def test_plugin_sandbox_passes_with_proof_artifact(monkeypatch, tmp_path):
    monkeypatch.delenv("PLUGIN_SANDBOX_RUNTIME_PROOF", raising=False)
    proof = tmp_path / "sandbox.txt"
    proof.write_text("network_blocked true", encoding="utf-8")

    assert checker.main(["--require", "plugin-sandbox", "--plugin-sandbox-proof", str(proof)]) == 0


def test_cloud_residency_evidence_path_is_required(monkeypatch, tmp_path):
    missing = tmp_path / "missing.json"
    monkeypatch.setenv("TRANSLATION_CLOUD_RESIDENCY_EVIDENCE", str(missing))

    assert checker.main(["--require", "cloud-residency"]) == 1

    evidence = tmp_path / "cloud-residency.json"
    evidence.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("TRANSLATION_CLOUD_RESIDENCY_EVIDENCE", str(evidence))
    assert checker.main(["--require", "cloud-residency"]) == 0
