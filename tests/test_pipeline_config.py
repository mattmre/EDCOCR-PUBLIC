"""Tests for :mod:`pipeline_config`.

Covers defaults, env-var parsing (int/float/bool), validation ranges, and
a smoke check that ``ocr_gpu_async`` can resolve the factory symbol.
"""

from __future__ import annotations

import importlib.util

import pytest

from pipeline_config import PipelineConfig, create_pipeline_config

# ---------------------------------------------------------------------------
# Defaults + env overrides
# ---------------------------------------------------------------------------


def test_default_config():
    """An empty env map should yield all default values."""
    cfg = create_pipeline_config({})

    # Group 1
    assert cfg.image_queue_size == 200
    assert cfg.chunk_queue_size == 50
    assert cfg.result_queue_size == 5000
    assert cfg.compression_queue_size == 5000

    # Group 2
    assert cfg.num_extractors == 8
    assert cfg.num_workers == 12
    assert cfg.num_compressors == 8
    assert cfg.num_assembler_workers == 4
    assert cfg.pdf_conversion_threads == 1
    assert cfg.extractor_process_workers == 8
    assert cfg.extractor_mode == "thread"

    # Group 3
    assert cfg.poppler_timeout == 300
    assert cfg.tesseract_timeout == 120
    assert cfg.thread_join_timeout == 30
    assert cfg.shutdown_drain_timeout_seconds == 300

    # Group 4
    assert cfg.dpi == 300
    assert cfg.jpeg_quality == 85
    assert cfg.monitor_sleep_seconds == 10
    assert cfg.keep_temp_files is False
    assert cfg.chunk_target_size == 20

    # Group 5
    assert cfg.video_frame_sample_seconds == 1.0
    assert cfg.video_max_frames == 300

    # Group 6
    assert cfg.fasttext_model_path == "/app/models/lid.176.bin"
    assert cfg.source_folder.replace("\\", "/").endswith("app/ocr_source")
    assert cfg.output_folder.replace("\\", "/").endswith("app/ocr_output")
    assert cfg.temp_folder.replace("\\", "/").endswith("app/ocr_temp")
    assert cfg.log_dir.replace("\\", "/").endswith("app/ocr_output/logs")
    assert cfg.failure_report.replace("\\", "/").endswith(
        "app/ocr_output/failures.csv"
    )
    assert cfg.healthcheck_file.replace("\\", "/").endswith("app/ocr_healthcheck")


def test_env_override_dpi():
    cfg = create_pipeline_config({"DPI": "450"})
    assert cfg.dpi == 450


def test_env_override_num_workers():
    cfg = create_pipeline_config({"NUM_WORKERS": "16"})
    assert cfg.num_workers == 16


def test_env_override_extractor_mode():
    cfg = create_pipeline_config({"EXTRACTOR_MODE": "process"})
    assert cfg.extractor_mode == "process"


def test_env_override_shutdown_drain_timeout():
    cfg = create_pipeline_config({"SHUTDOWN_DRAIN_TIMEOUT": "123"})
    assert cfg.shutdown_drain_timeout_seconds == 123


# ---------------------------------------------------------------------------
# Bool parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "True", "yes", "YES", "Yes"])
def test_bool_keep_temp_files_true(value):
    """Truthy strings (case-insensitive) must enable the flag."""
    cfg = create_pipeline_config({"KEEP_TEMP_FILES": value})
    assert cfg.keep_temp_files is True


@pytest.mark.parametrize("value", ["", "false", "FALSE", "0", "no", "off"])
def test_bool_keep_temp_files_false(value):
    """Empty / falsy values must leave the flag disabled."""
    cfg = create_pipeline_config({"KEEP_TEMP_FILES": value})
    assert cfg.keep_temp_files is False


# ---------------------------------------------------------------------------
# Validation ranges
# ---------------------------------------------------------------------------


def test_validation_dpi_too_low():
    with pytest.raises(ValueError, match="DPI"):
        PipelineConfig(dpi=10)


def test_validation_dpi_too_high():
    with pytest.raises(ValueError, match="DPI"):
        PipelineConfig(dpi=9999)


def test_validation_jpeg_quality_out_of_range():
    with pytest.raises(ValueError, match="jpeg_quality"):
        PipelineConfig(jpeg_quality=0)
    with pytest.raises(ValueError, match="jpeg_quality"):
        PipelineConfig(jpeg_quality=101)


def test_validation_num_workers_zero():
    with pytest.raises(ValueError, match="num_workers"):
        PipelineConfig(num_workers=0)


def test_validation_extractor_mode_invalid():
    with pytest.raises(ValueError, match="extractor_mode"):
        PipelineConfig(extractor_mode="bogus")


# ---------------------------------------------------------------------------
# Repr + misc fields
# ---------------------------------------------------------------------------


def test_repr_contains_groups():
    cfg = create_pipeline_config({})
    rendered = repr(cfg)
    assert "Queue" in rendered
    assert "Threads" in rendered
    assert "Timeouts" in rendered
    assert "Processing" in rendered
    assert "Video" in rendered
    assert "ML + runtime paths" in rendered


def test_video_frame_sample_seconds():
    cfg = create_pipeline_config({"VIDEO_FRAME_SAMPLE_SECONDS": "2.5"})
    assert cfg.video_frame_sample_seconds == 2.5


def test_fasttext_model_path_override():
    cfg = create_pipeline_config({"FASTTEXT_MODEL_PATH": "/tmp/lid.bin"})
    assert cfg.fasttext_model_path == "/tmp/lid.bin"


def test_invalid_output_folder_falls_back_to_default():
    cfg = create_pipeline_config({"OUTPUT_FOLDER": "/tmp/not-allowed"})
    assert cfg.output_folder.replace("\\", "/").endswith("app/ocr_output")


# ---------------------------------------------------------------------------
# Integration smoke: ocr_gpu_async exposes create_pipeline_config
# ---------------------------------------------------------------------------


def test_pipeline_config_import_in_ocr_async():
    """Smoke-check that the ``ocr_gpu_async`` module references the factory.

    We deliberately avoid ``import ocr_gpu_async`` here because the module
    pulls in heavy optional dependencies (PaddleOCR, PyMuPDF, fasttext)
    that may be unavailable in the CI environment.  Scanning the source
    file is sufficient to prove the Phase 1 wiring is in place.
    """
    spec = importlib.util.find_spec("ocr_gpu_async")
    assert spec is not None, "ocr_gpu_async module should be discoverable"
    assert spec.origin is not None

    with open(spec.origin, encoding="utf-8") as fh:
        source = fh.read()

    assert "create_pipeline_config" in source and "PipelineConfig" in source, (
        "ocr_gpu_async must import PipelineConfig and create_pipeline_config"
    )
    assert "_ACTIVE_PIPELINE_CONFIG = create_pipeline_config()" in source, (
        "ocr_gpu_async must create the active PipelineConfig at module import"
    )


# ===========================================================================
# Phase 2 -- feature flag fields (Groups 7-12)
# ===========================================================================


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_default_config_feature_flags():
    """Empty env should yield Phase 2 defaults from the spec."""
    cfg = create_pipeline_config({})

    # Group 7: Document Intelligence
    assert cfg.enable_document_intelligence is False
    assert cfg.enable_layout_analysis is True
    assert cfg.enable_table_extraction is True
    assert cfg.docintel_mode == "full"
    assert cfg.export_tables is False

    # Group 8: Form & KV
    assert cfg.enable_form_detection is False
    assert cfg.enable_kv_extraction is False
    assert cfg.enable_privilege_detection is False

    # Group 9: Custody
    assert cfg.enable_custody is True

    # Group 10: Processing features
    assert cfg.enable_preprocessing is False
    assert cfg.preprocessing_level == "standard"
    assert cfg.enable_noise_profiling is False
    assert cfg.enable_validation is True
    assert cfg.enable_dpi_escalation is False
    assert cfg.dpi_confidence_threshold == 0.60

    # Group 11: Analysis features
    assert cfg.enable_ner is False
    assert cfg.enable_handwriting is False
    assert cfg.enable_signature_verification is False
    assert cfg.enable_vertical_text is False
    assert cfg.enable_table_fallback is False
    assert cfg.enable_classification is False
    assert cfg.enable_extraction is False
    assert cfg.enable_specialist_routing is False
    assert cfg.enable_entity_consolidation is False
    assert cfg.enable_relationship_extraction is False
    assert cfg.enable_retrieval_output is False
    assert cfg.enable_exception_routing is False

    # Group 12: Pipeline optimization
    assert cfg.enable_adaptive_batch is False
    assert cfg.enable_page_cache is False
    assert cfg.enable_page_routing is False
    assert cfg.enable_gpu_optimization is False


def test_enable_custody_default_true():
    """`enable_custody` is one of only two boolean defaults that start True."""
    cfg = create_pipeline_config({})
    assert cfg.enable_custody is True


def test_enable_validation_default_true():
    """`enable_validation` defaults to True (forensic safety)."""
    cfg = create_pipeline_config({})
    assert cfg.enable_validation is True


def test_enable_layout_and_table_default_true():
    """Document Intelligence sub-toggles default True (gated by master flag)."""
    cfg = create_pipeline_config({})
    assert cfg.enable_layout_analysis is True
    assert cfg.enable_table_extraction is True


# ---------------------------------------------------------------------------
# Env override -- single boolean flags
# ---------------------------------------------------------------------------


def test_enable_ner_true():
    cfg = create_pipeline_config({"ENABLE_NER": "1"})
    assert cfg.enable_ner is True


def test_enable_ner_false():
    cfg = create_pipeline_config({"ENABLE_NER": "0"})
    assert cfg.enable_ner is False


def test_enable_validation_explicit_false():
    cfg = create_pipeline_config({"ENABLE_VALIDATION": "false"})
    assert cfg.enable_validation is False


def test_enable_custody_explicit_false():
    cfg = create_pipeline_config({"ENABLE_CUSTODY": "no"})
    assert cfg.enable_custody is False


def test_enable_document_intelligence_true():
    cfg = create_pipeline_config({"ENABLE_DOCUMENT_INTELLIGENCE": "yes"})
    assert cfg.enable_document_intelligence is True


def test_enable_handwriting_true():
    cfg = create_pipeline_config({"ENABLE_HANDWRITING": "true"})
    assert cfg.enable_handwriting is True


def test_enable_signature_verification_true():
    cfg = create_pipeline_config({"ENABLE_SIGNATURE_VERIFICATION": "1"})
    assert cfg.enable_signature_verification is True


def test_enable_vertical_text_true():
    cfg = create_pipeline_config({"ENABLE_VERTICAL_TEXT": "yes"})
    assert cfg.enable_vertical_text is True


def test_enable_classification_true():
    cfg = create_pipeline_config({"ENABLE_CLASSIFICATION": "true"})
    assert cfg.enable_classification is True


def test_enable_adaptive_batch_true():
    cfg = create_pipeline_config({"ENABLE_ADAPTIVE_BATCH": "1"})
    assert cfg.enable_adaptive_batch is True


def test_enable_gpu_optimization_true():
    cfg = create_pipeline_config({"ENABLE_GPU_OPTIMIZATION": "true"})
    assert cfg.enable_gpu_optimization is True


def test_export_tables_true():
    cfg = create_pipeline_config({"EXPORT_TABLES": "yes"})
    assert cfg.export_tables is True


# Parametrized broad-coverage of all 30 boolean fields
_BOOL_FIELDS = [
    ("ENABLE_DOCUMENT_INTELLIGENCE", "enable_document_intelligence", False),
    ("ENABLE_LAYOUT_ANALYSIS", "enable_layout_analysis", True),
    ("ENABLE_TABLE_EXTRACTION", "enable_table_extraction", True),
    ("EXPORT_TABLES", "export_tables", False),
    ("ENABLE_FORM_DETECTION", "enable_form_detection", False),
    ("ENABLE_KV_EXTRACTION", "enable_kv_extraction", False),
    ("ENABLE_PRIVILEGE_DETECTION", "enable_privilege_detection", False),
    ("ENABLE_CUSTODY", "enable_custody", True),
    ("ENABLE_PREPROCESSING", "enable_preprocessing", False),
    ("ENABLE_NOISE_PROFILING", "enable_noise_profiling", False),
    ("ENABLE_VALIDATION", "enable_validation", True),
    ("ENABLE_DPI_ESCALATION", "enable_dpi_escalation", False),
    ("ENABLE_NER", "enable_ner", False),
    ("ENABLE_HANDWRITING", "enable_handwriting", False),
    ("ENABLE_SIGNATURE_VERIFICATION", "enable_signature_verification", False),
    ("ENABLE_VERTICAL_TEXT", "enable_vertical_text", False),
    ("ENABLE_TABLE_FALLBACK", "enable_table_fallback", False),
    ("ENABLE_CLASSIFICATION", "enable_classification", False),
    ("ENABLE_EXTRACTION", "enable_extraction", False),
    ("ENABLE_SPECIALIST_ROUTING", "enable_specialist_routing", False),
    ("ENABLE_ENTITY_CONSOLIDATION", "enable_entity_consolidation", False),
    ("ENABLE_RELATIONSHIP_EXTRACTION", "enable_relationship_extraction", False),
    ("ENABLE_RETRIEVAL_OUTPUT", "enable_retrieval_output", False),
    ("ENABLE_EXCEPTION_ROUTING", "enable_exception_routing", False),
    ("ENABLE_ADAPTIVE_BATCH", "enable_adaptive_batch", False),
    ("ENABLE_PAGE_CACHE", "enable_page_cache", False),
    ("ENABLE_PAGE_ROUTING", "enable_page_routing", False),
    ("ENABLE_GPU_OPTIMIZATION", "enable_gpu_optimization", False),
]


@pytest.mark.parametrize("env_key,attr,default", _BOOL_FIELDS)
def test_all_bool_flags_truthy_override(env_key, attr, default):
    """Every boolean field accepts a truthy override."""
    cfg = create_pipeline_config({env_key: "1"})
    assert getattr(cfg, attr) is True


@pytest.mark.parametrize("env_key,attr,default", _BOOL_FIELDS)
def test_all_bool_flags_falsy_override(env_key, attr, default):
    """Every boolean field accepts a falsy override."""
    cfg = create_pipeline_config({env_key: "0"})
    assert getattr(cfg, attr) is False


@pytest.mark.parametrize("env_key,attr,default", _BOOL_FIELDS)
def test_all_bool_flags_default(env_key, attr, default):
    """Every boolean field falls back to its declared default."""
    cfg = create_pipeline_config({})
    assert getattr(cfg, attr) is default


# ---------------------------------------------------------------------------
# String-choice fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["layout_only", "tables_only", "full"])
def test_docintel_mode_valid(mode):
    cfg = create_pipeline_config({"DOCINTEL_MODE": mode})
    assert cfg.docintel_mode == mode


def test_docintel_mode_uppercase_normalized():
    """Case-insensitive parsing of choice values."""
    cfg = create_pipeline_config({"DOCINTEL_MODE": "TABLES_ONLY"})
    assert cfg.docintel_mode == "tables_only"


def test_docintel_mode_invalid_falls_back_to_default():
    """Unknown values silently fall back to the default ('full')."""
    cfg = create_pipeline_config({"DOCINTEL_MODE": "invalid"})
    assert cfg.docintel_mode == "full"


def test_docintel_mode_empty_falls_back_to_default():
    cfg = create_pipeline_config({"DOCINTEL_MODE": ""})
    assert cfg.docintel_mode == "full"


@pytest.mark.parametrize(
    "level", ["none", "standard", "enhanced", "aggressive"]
)
def test_preprocessing_level_valid(level):
    cfg = create_pipeline_config({"PREPROCESSING_LEVEL": level})
    assert cfg.preprocessing_level == level


def test_preprocessing_level_invalid_falls_back_to_default():
    cfg = create_pipeline_config({"PREPROCESSING_LEVEL": "ultra"})
    assert cfg.preprocessing_level == "standard"


def test_preprocessing_level_uppercase_normalized():
    cfg = create_pipeline_config({"PREPROCESSING_LEVEL": "Aggressive"})
    assert cfg.preprocessing_level == "aggressive"


# ---------------------------------------------------------------------------
# Float field: dpi_confidence_threshold
# ---------------------------------------------------------------------------


def test_dpi_confidence_threshold_default():
    cfg = create_pipeline_config({})
    assert cfg.dpi_confidence_threshold == 0.60


def test_dpi_confidence_threshold_override():
    cfg = create_pipeline_config({"DPI_CONFIDENCE_THRESHOLD": "0.75"})
    assert cfg.dpi_confidence_threshold == 0.75


def test_dpi_confidence_threshold_clamps_above_range():
    """Out-of-range values clamp to max (1.0) -- matches `_float` helper."""
    cfg = create_pipeline_config({"DPI_CONFIDENCE_THRESHOLD": "2.0"})
    assert cfg.dpi_confidence_threshold == 1.0


def test_dpi_confidence_threshold_clamps_below_range():
    """Negative values clamp to min (0.0)."""
    cfg = create_pipeline_config({"DPI_CONFIDENCE_THRESHOLD": "-0.5"})
    assert cfg.dpi_confidence_threshold == 0.0


def test_dpi_confidence_threshold_invalid_falls_back():
    """Non-numeric strings fall back to the declared default."""
    cfg = create_pipeline_config({"DPI_CONFIDENCE_THRESHOLD": "not-a-float"})
    assert cfg.dpi_confidence_threshold == 0.60


# ---------------------------------------------------------------------------
# Validation failures (direct construction)
# ---------------------------------------------------------------------------


def test_invalid_docintel_mode_raises():
    with pytest.raises(ValueError, match="docintel_mode"):
        PipelineConfig(docintel_mode="bad")


def test_invalid_preprocessing_level_raises():
    with pytest.raises(ValueError, match="preprocessing_level"):
        PipelineConfig(preprocessing_level="ultra")


def test_invalid_dpi_confidence_high_raises():
    with pytest.raises(ValueError, match="dpi_confidence_threshold"):
        PipelineConfig(dpi_confidence_threshold=1.5)


def test_invalid_dpi_confidence_negative_raises():
    with pytest.raises(ValueError, match="dpi_confidence_threshold"):
        PipelineConfig(dpi_confidence_threshold=-0.1)


def test_invalid_dpi_confidence_type_raises():
    with pytest.raises(ValueError, match="dpi_confidence_threshold"):
        PipelineConfig(dpi_confidence_threshold="high")  # type: ignore[arg-type]


def test_invalid_enable_ner_type_raises():
    with pytest.raises(ValueError, match="enable_ner"):
        PipelineConfig(enable_ner="yes")  # type: ignore[arg-type]


def test_invalid_enable_custody_type_raises():
    with pytest.raises(ValueError, match="enable_custody"):
        PipelineConfig(enable_custody=1)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Repr coverage for Phase 2 groups
# ---------------------------------------------------------------------------


def test_repr_contains_phase2_groups():
    cfg = create_pipeline_config({})
    rendered = repr(cfg)
    assert "Document Intelligence" in rendered
    assert "Form & KV" in rendered
    assert "Custody" in rendered
    assert "Processing features" in rendered
    assert "Analysis features" in rendered
    assert "Pipeline optimization" in rendered
    assert "enable_ner" in rendered
    assert "docintel_mode" in rendered
    assert "preprocessing_level" in rendered
    assert "dpi_confidence_threshold" in rendered
