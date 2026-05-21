from __future__ import annotations

from pathlib import Path

from scripts import run_translation_e2e_harness as harness


def test_inventory_contains_required_translation_surfaces():
    inventory = harness.inventory_api_surface()
    backend_paths = {row["path"] for row in inventory["backend"]}
    assert "/api/v1/translation/jobs" in backend_paths
    assert "/api/v1/translation/score-pair" in backend_paths
    assert "/api/v1/translation/batches" in backend_paths
    assert "/api/v1/translation/tenants/{tenant_id}/config" in backend_paths
    assert "/api/v1/review/{id}/certify" in backend_paths
    assert inventory["frontend"]


def test_minimal_stack_reports_model_dir(tmp_path):
    model_dir = tmp_path / "model"
    stack = harness.minimal_runnable_stack(model_dir)
    assert stack["translation_engine"]["candidate"] == "local_ct2_opus"
    assert stack["translation_engine"]["model_dir"] == str(model_dir)
    assert stack["translation_engine"]["model_dir_exists"] is False


def test_endpoint_matrix_mentions_contract_execution():
    matrix = harness.endpoint_matrix_md(
        harness.inventory_api_surface(),
        {"status": "passed", "requests_executed": 6},
    )
    assert "Endpoint Coverage Matrix" in matrix
    assert "`/api/v1/translation/jobs`" in matrix
    assert "Requests executed: `6`" in matrix


def test_evidence_markdown_keeps_scores_separate(tmp_path):
    product = {
        "classification": "local_e2e_pass",
        "status": "passed",
        "engine": {"mode": "real_ct2"},
        "custody": {"chain_file_verified": True},
        "sidecars": {"json": "x"},
        "not_production_live": True,
    }
    text = harness.evidence_md(
        harness.minimal_runnable_stack(Path("missing")),
        {"status": "passed", "requests_executed": 6},
        product,
        tmp_path,
    )
    assert "Existing local remediation release score remains **99/100**" in text
    assert "Product E2E score for this lane" in text
    assert "not production/live evidence" in text


def test_product_path_runs_with_stub_engine(tmp_path):
    result = harness.run_product_path(
        tmp_path,
        tmp_path / "missing-model",
        allow_stub_engine=True,
    )
    assert result["status"] == "partial"
    assert result["engine"]["mode"] == "stub"
    assert result["custody"]["chain_file_verified"] is True
    assert result["language_detection"]["page_count"] == 2
    assert result["sidecars"]["json"].endswith(".translation.json")
    assert result["skipped_output"][0]["engine_id"] == "language_gate_already_target"


def test_clean_previous_evidence_preserves_run_log(tmp_path):
    (tmp_path / "tsa").mkdir()
    (tmp_path / "tsa" / "old.tsr").write_text("old", encoding="utf-8")
    (tmp_path / "product-path-result.json").write_text("{}", encoding="utf-8")
    (tmp_path / "e2e-harness-run.txt").write_text("keep", encoding="utf-8")

    harness.clean_previous_evidence(tmp_path)

    assert not (tmp_path / "tsa").exists()
    assert not (tmp_path / "product-path-result.json").exists()
    assert (tmp_path / "e2e-harness-run.txt").read_text(encoding="utf-8") == "keep"
