"""Runtime wiring tests for  Phase 3."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from pipeline_config import create_pipeline_config


def _make_args(**overrides):
    base = {
        "enable_docintel": False,
        "docintel_mode": "full",
        "export_tables": False,
        "enable_form_detection": False,
        "enable_kv_extraction": False,
        "enable_privilege_detection": False,
        "no_custody": False,
        "enable_preprocessing": False,
        "preprocessing_level": "standard",
        "enable_noise_profiling": False,
        "enable_dpi_escalation": False,
        "enable_ner": False,
        "enable_handwriting": False,
        "enable_signature_verification": False,
        "enable_vertical_text": False,
        "enable_table_fallback": False,
        "enable_classification": False,
        "enable_extraction": False,
        "enable_specialist_routing": False,
        "enable_entity_consolidation": False,
        "enable_relationship_extraction": False,
        "enable_retrieval_output": False,
        "enable_exception_routing": False,
        "extractor_mode": None,
        "extractor_process_workers": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture(scope="module")
def pipe():
    return importlib.import_module("ocr_gpu_async")


def test_apply_cli_overrides_mutates_config(pipe):
    cfg = create_pipeline_config({})
    args = _make_args(
        enable_docintel=True,
        docintel_mode="tables_only",
        export_tables=True,
        enable_form_detection=True,
        enable_ner=True,
        enable_preprocessing=True,
        preprocessing_level="enhanced",
        extractor_mode="process",
        extractor_process_workers=5,
    )

    pipe._apply_cli_overrides(args, cfg)

    assert cfg.enable_document_intelligence is True
    assert cfg.docintel_mode == "tables_only"
    assert cfg.export_tables is True
    assert cfg.enable_form_detection is True
    assert cfg.enable_ner is True
    assert cfg.enable_preprocessing is True
    assert cfg.preprocessing_level == "enhanced"
    assert cfg.extractor_mode == "process"
    assert cfg.extractor_process_workers == 5


def test_apply_cli_overrides_requires_docintel_for_form_flags(pipe):
    cfg = create_pipeline_config({})
    args = _make_args(enable_form_detection=True, enable_kv_extraction=True)

    pipe._apply_cli_overrides(args, cfg)

    assert cfg.enable_document_intelligence is False
    assert cfg.enable_form_detection is False
    assert cfg.enable_kv_extraction is False


def test_activate_pipeline_config_syncs_runtime_globals(pipe):
    original_cfg = pipe._ACTIVE_PIPELINE_CONFIG
    cfg = create_pipeline_config(
        {
            "NUM_WORKERS": "7",
            "ENABLE_NER": "1",
            "EXTRACTOR_MODE": "process",
            "EXTRACTOR_PROCESS_WORKERS": "3",
            "SHUTDOWN_DRAIN_TIMEOUT": "123",
        }
    )

    try:
        pipe._activate_pipeline_config(cfg)

        assert pipe._ACTIVE_PIPELINE_CONFIG is cfg
        assert pipe.NUM_WORKERS == 7
        assert pipe.ENABLE_NER is True
        assert pipe.EXTRACTOR_MODE == "process"
        assert pipe.EXTRACTOR_PROCESS_WORKERS == 3
        assert pipe.SHUTDOWN_DRAIN_TIMEOUT_SECONDS == 123
    finally:
        pipe._activate_pipeline_config(original_cfg)
