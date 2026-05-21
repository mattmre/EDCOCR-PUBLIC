"""Tests for the ocr_local namespace compatibility layer.

These tests verify that:
1. ocr_local sub-packages are importable
2. Sub-package attribute access works (e.g. ocr_local.features.ner)
3. Direct symbol import works (e.g. from ocr_local.features.ner import NERExtractor)
4. Original root-level imports are not broken
5. Phase 2 migration terminal state: every lazy-loader tuple and
   compat frozenset is empty, and every sub-package module resolves
   through the standard import machinery

Some feature modules pull in heavy optional dependencies (spaCy, paddleocr,
torch, etc.).  We guard the import-triggering tests with
``importlib.util.find_spec`` so the suite stays fast and green in slim
environments.
"""

from __future__ import annotations

import importlib
import importlib.util

import pytest


def _module_available(name: str) -> bool:
    """Return True if ``name`` can be located without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


# ---------------------------------------------------------------------------
# 1. Package importability
# ---------------------------------------------------------------------------


def test_ocr_local_package_importable() -> None:
    import ocr_local  # noqa: F401

    assert hasattr(ocr_local, "__version__")


def test_features_subpackage_importable() -> None:
    import ocr_local.features  # noqa: F401


def test_ml_subpackage_importable() -> None:
    import ocr_local.ml  # noqa: F401


def test_infra_subpackage_importable() -> None:
    import ocr_local.infra  # noqa: F401


def test_config_subpackage_importable() -> None:
    import ocr_local.config  # noqa: F401


# ---------------------------------------------------------------------------
# 2. Lazy attribute access
# ---------------------------------------------------------------------------


def test_phase2_migration_complete() -> None:
    """ Phase 2 terminal state -- all sub-packages fully migrated.

    Every compat sub-package frozenset in ``_SUBPACKAGE_MODULES`` must be
    empty, every lazy-loader tuple must be an empty tuple, and each
    canonical sub-package must be importable through the standard
    import machinery (no meta-path trickery required).
    """
    from ocr_local._compat_finder import _SUBPACKAGE_MODULES
    from ocr_local.config import _CONFIG_MODULES
    from ocr_local.features import _FEATURE_MODULES
    from ocr_local.infra import _INFRA_MODULES
    from ocr_local.ml import _ML_MODULES

    # 1. All lazy-loader tuples are empty.
    assert _FEATURE_MODULES == ()
    assert _ML_MODULES == ()
    assert _INFRA_MODULES == ()
    assert _CONFIG_MODULES == ()

    # 2. All compat frozensets are empty.
    expected_subpackages = {
        "ocr_local.features",
        "ocr_local.ml",
        "ocr_local.infra",
        "ocr_local.config",
    }
    assert set(_SUBPACKAGE_MODULES) == expected_subpackages
    for key in expected_subpackages:
        assert _SUBPACKAGE_MODULES[key] == frozenset(), key

    # 3. Every canonical sub-package imports natively.
    for dotted in expected_subpackages:
        pkg = importlib.import_module(dotted)
        assert pkg.__name__ == dotted


def test_config_module_lazy_access() -> None:
    if not _module_available("ocr_local.config.language_config"):
        pytest.skip("language_config module not available in this environment")
    from ocr_local import config

    mod = config.language_config
    assert hasattr(mod, "__name__")
    assert mod.__name__ == "ocr_local.config.language_config"


# ---------------------------------------------------------------------------
# 3. Direct dotted imports
# ---------------------------------------------------------------------------


def test_config_direct_import() -> None:
    if not _module_available("ocr_local.config.version"):
        pytest.skip("version module not available in this environment")
    import ocr_local.config.version as version_mod

    assert version_mod.__name__ == "ocr_local.config.version"


# ---------------------------------------------------------------------------
# 5. Negative cases
# ---------------------------------------------------------------------------


def test_invalid_feature_attr_raises() -> None:
    import ocr_local.features as features

    with pytest.raises(AttributeError):
        _ = features.does_not_exist


# ---------------------------------------------------------------------------
# 6. Version sentinel
# ---------------------------------------------------------------------------


def test_ocr_local_version() -> None:
    import ocr_local

    assert ocr_local.__version__ == "1.0.0-compat"


# ---------------------------------------------------------------------------
# 7.  Phase 2: physically migrated modules
# ---------------------------------------------------------------------------


def test_noise_profiling_native_subpackage_import() -> None:
    """After Phase 2 migration, ocr_local.features.noise_profiling is native."""
    mod = importlib.import_module("ocr_local.features.noise_profiling")
    # Native subpackage modules have their own __name__, not the root name
    assert mod.__name__ == "ocr_local.features.noise_profiling"
    assert hasattr(mod, "NoiseProfile")
    assert hasattr(mod, "profile_image")


def test_noise_profiling_root_shim_still_works() -> None:
    """Root-level 'import noise_profiling' continues to work via the shim."""
    import noise_profiling  # noqa: F401

    assert hasattr(noise_profiling, "NoiseProfile")
    assert hasattr(noise_profiling, "profile_image")
    assert hasattr(noise_profiling, "_NOISE_PROFILING_AVAILABLE")
    assert hasattr(noise_profiling, "ENABLE_NOISE_PROFILING")


def test_noise_profiling_root_shim_private_symbols() -> None:
    """Private symbols (_xxx) are accessible via the root shim."""
    import noise_profiling  # noqa: F401

    # These are imported by tests and by preprocessing.py
    assert hasattr(noise_profiling, "_NOISE_PROFILING_AVAILABLE")
    assert hasattr(noise_profiling, "_default_profile")


def test_noise_profiling_from_import() -> None:
    """'from noise_profiling import X' works for all public symbols."""
    from noise_profiling import NoiseProfile, estimate_noise_variance, profile_image

    assert callable(profile_image)
    assert callable(estimate_noise_variance)
    assert isinstance(NoiseProfile.__doc__, str) or True  # dataclass


def test_noise_profiling_not_in_feature_modules() -> None:
    """After Phase 2 migration, noise_profiling is not in the lazy-loader list."""
    from ocr_local.features import _FEATURE_MODULES  # noqa: F401

    assert "noise_profiling" not in _FEATURE_MODULES


def test_noise_profiling_not_in_compat_finder() -> None:
    """After Phase 2 migration, the compat finder no longer intercepts noise_profiling."""
    from ocr_local._compat_finder import _SUBPACKAGE_MODULES

    assert "noise_profiling" not in _SUBPACKAGE_MODULES["ocr_local.features"]


def test_noise_profiling_feature_attr_access() -> None:
    """ocr_local.features.noise_profiling is accessible as a package attribute."""
    import ocr_local.features as features
    import ocr_local.features.noise_profiling  # triggers subpackage load  # noqa: F401

    mod = features.noise_profiling
    assert mod.__name__ == "ocr_local.features.noise_profiling"


# ---------------------------------------------------------------------------
# sla_monitoring -- Phase 2 migration tests
# ---------------------------------------------------------------------------


def test_sla_monitoring_native_subpackage_import() -> None:
    """After Phase 2 migration, ocr_local.features.sla_monitoring is native."""
    mod = importlib.import_module("ocr_local.features.sla_monitoring")
    assert mod.__file__ is not None
    assert "ocr_local" in mod.__file__ or "ocr_local" in mod.__name__
    assert mod.__name__ == "ocr_local.features.sla_monitoring"


def test_sla_monitoring_root_shim_still_works() -> None:
    """Root-level 'import sla_monitoring' continues to work via the shim."""
    import sla_monitoring  # noqa: F401

    assert hasattr(sla_monitoring, "SLAMonitor")
    assert hasattr(sla_monitoring, "SLAReport")


def test_sla_monitoring_root_shim_private_symbols() -> None:
    """Private symbols are accessible via the root shim."""
    import sla_monitoring  # noqa: F401

    # The shim replaces sys.modules so private attrs come from the real module
    assert hasattr(sla_monitoring, "_SEMVER_RE")


def test_sla_monitoring_not_in_feature_modules() -> None:
    """After Phase 2 migration, sla_monitoring is not in the lazy-loader list."""
    from ocr_local.features import _FEATURE_MODULES

    assert "sla_monitoring" not in _FEATURE_MODULES


def test_sla_monitoring_not_in_compat_finder() -> None:
    """After Phase 2 migration, the compat finder no longer intercepts sla_monitoring."""
    from ocr_local._compat_finder import _SUBPACKAGE_MODULES

    assert "sla_monitoring" not in _SUBPACKAGE_MODULES["ocr_local.features"]


def test_sla_monitoring_feature_attr_access() -> None:
    """ocr_local.features.sla_monitoring is accessible as a package attribute."""
    import ocr_local.features as features
    import ocr_local.features.sla_monitoring  # noqa: F401

    mod = features.sla_monitoring
    assert mod.__name__ == "ocr_local.features.sla_monitoring"


# ---------------------------------------------------------------------------
# validation -- Phase 2 migration tests
# ---------------------------------------------------------------------------


def test_validation_native_subpackage_import() -> None:
    """After Phase 2 migration, ocr_local.features.validation is native."""
    mod = importlib.import_module("ocr_local.features.validation")
    assert mod.__file__ is not None
    assert "ocr_local" in mod.__file__ or "ocr_local" in mod.__name__
    assert mod.__name__ == "ocr_local.features.validation"


def test_validation_root_shim_still_works() -> None:
    """Root-level 'import validation' continues to work via the shim."""
    import validation  # noqa: F401

    assert hasattr(validation, "PageValidation")
    assert hasattr(validation, "DocumentValidation")
    assert hasattr(validation, "classify_quality")
    assert hasattr(validation, "finalize_validation")
    assert hasattr(validation, "write_validation_json")


def test_validation_root_shim_private_symbols() -> None:
    """Non-``__all__`` module attributes are accessible via the root shim."""
    import validation  # noqa: F401

    # ``logger`` is a module-level attribute that is not declared in
    # ``__all__``; the sys.modules-replacement shim must still expose it.
    assert hasattr(validation, "logger")


def test_validation_not_in_feature_modules() -> None:
    """After Phase 2 migration, validation is not in the lazy-loader list."""
    from ocr_local.features import _FEATURE_MODULES

    assert "validation" not in _FEATURE_MODULES


def test_validation_not_in_compat_finder() -> None:
    """After Phase 2 migration, the compat finder no longer intercepts validation."""
    from ocr_local._compat_finder import _SUBPACKAGE_MODULES

    assert "validation" not in _SUBPACKAGE_MODULES["ocr_local.features"]


def test_validation_feature_attr_access() -> None:
    """ocr_local.features.validation is accessible as a package attribute."""
    import ocr_local.features as features
    import ocr_local.features.validation  # noqa: F401

    mod = features.validation
    assert mod.__name__ == "ocr_local.features.validation"


# ---------------------------------------------------------------------------
# dpi_escalation -- Phase 2 migration tests
# ---------------------------------------------------------------------------


def test_dpi_escalation_native_subpackage_import() -> None:
    """After Phase 2 migration, ocr_local.features.dpi_escalation is native."""
    mod = importlib.import_module("ocr_local.features.dpi_escalation")
    assert mod.__file__ is not None
    assert "ocr_local" in mod.__file__ or "ocr_local" in mod.__name__
    assert mod.__name__ == "ocr_local.features.dpi_escalation"


def test_dpi_escalation_root_shim_still_works() -> None:
    """Root-level 'import dpi_escalation' continues to work via the shim."""
    import dpi_escalation  # noqa: F401

    assert hasattr(dpi_escalation, "should_escalate")
    assert hasattr(dpi_escalation, "get_next_dpi")
    assert hasattr(dpi_escalation, "re_extract_page_at_dpi")
    assert hasattr(dpi_escalation, "EscalationResult")
    assert hasattr(dpi_escalation, "DPI_SCHEDULE")


def test_dpi_escalation_root_shim_private_symbols() -> None:
    """Private symbols are accessible via the root shim."""
    import dpi_escalation  # noqa: F401

    # ``_convert_from_path`` is a module-level private import that is not
    # declared in ``__all__``; the sys.modules-replacement shim must still
    # expose it.  ``logger`` is similarly a non-``__all__`` attribute.
    assert hasattr(dpi_escalation, "_convert_from_path")
    assert hasattr(dpi_escalation, "logger")


def test_dpi_escalation_not_in_feature_modules() -> None:
    """After Phase 2 migration, dpi_escalation is not in the lazy-loader list."""
    from ocr_local.features import _FEATURE_MODULES

    assert "dpi_escalation" not in _FEATURE_MODULES


def test_dpi_escalation_not_in_compat_finder() -> None:
    """After Phase 2 migration, the compat finder no longer intercepts dpi_escalation."""
    from ocr_local._compat_finder import _SUBPACKAGE_MODULES

    assert "dpi_escalation" not in _SUBPACKAGE_MODULES["ocr_local.features"]


def test_dpi_escalation_feature_attr_access() -> None:
    """ocr_local.features.dpi_escalation is accessible as a package attribute."""
    import ocr_local.features as features
    import ocr_local.features.dpi_escalation  # noqa: F401

    mod = features.dpi_escalation
    assert mod.__name__ == "ocr_local.features.dpi_escalation"


# ---------------------------------------------------------------------------
# handwriting -- Phase 2 migration tests
# ---------------------------------------------------------------------------


def test_handwriting_native_subpackage_import() -> None:
    """After Phase 2 migration, ocr_local.features.handwriting is native."""
    mod = importlib.import_module("ocr_local.features.handwriting")
    assert mod.__file__ is not None
    assert mod.__name__ == "ocr_local.features.handwriting"


def test_handwriting_root_shim_still_works() -> None:
    """Root-level 'import handwriting' continues to work via the shim."""
    import handwriting  # noqa: F401

    assert hasattr(handwriting, "HandwritingRegion")
    assert hasattr(handwriting, "PageHandwriting")
    assert hasattr(handwriting, "DocumentHandwriting")
    assert hasattr(handwriting, "detect_handwriting_by_confidence")
    assert hasattr(handwriting, "merge_handwriting_signals")
    assert hasattr(handwriting, "finalize_handwriting")
    assert hasattr(handwriting, "write_handwriting_json")


def test_handwriting_root_shim_private_symbols() -> None:
    """Private symbols are accessible via the root shim."""
    import handwriting  # noqa: F401

    # ``_WEIGHT_CONFIDENCE`` is a module-level private constant not in
    # ``__all__``; the sys.modules-replacement shim must still expose it.
    # ``_coefficient_of_variation`` is a private helper; ``logger`` and
    # ``_CV2_AVAILABLE`` are likewise non-``__all__`` attributes.
    assert hasattr(handwriting, "_WEIGHT_CONFIDENCE")
    assert hasattr(handwriting, "_coefficient_of_variation")
    assert hasattr(handwriting, "_CV2_AVAILABLE")
    assert hasattr(handwriting, "logger")


def test_handwriting_not_in_feature_modules() -> None:
    """After Phase 2 migration, handwriting is not in the lazy-loader list."""
    from ocr_local.features import _FEATURE_MODULES

    assert "handwriting" not in _FEATURE_MODULES


def test_handwriting_not_in_compat_finder() -> None:
    """After Phase 2 migration, the compat finder no longer intercepts handwriting."""
    from ocr_local._compat_finder import _SUBPACKAGE_MODULES

    assert "handwriting" not in _SUBPACKAGE_MODULES["ocr_local.features"]


def test_handwriting_feature_attr_access() -> None:
    """ocr_local.features.handwriting is accessible as a package attribute."""
    import ocr_local.features as features
    import ocr_local.features.handwriting  # noqa: F401

    mod = features.handwriting
    assert mod.__name__ == "ocr_local.features.handwriting"


# ---------------------------------------------------------------------------
# ner -- Phase 2 migration tests
# ---------------------------------------------------------------------------


def test_ner_native_subpackage_import() -> None:
    """After Phase 2 migration, ocr_local.features.ner is native."""
    mod = importlib.import_module("ocr_local.features.ner")
    assert mod.__file__ is not None
    assert mod.__name__ == "ocr_local.features.ner"


def test_ner_root_shim_still_works() -> None:
    """Root-level 'import ner' continues to work via the shim."""
    import ner  # noqa: F401

    assert hasattr(ner, "Entity")
    assert hasattr(ner, "PageNER")
    assert hasattr(ner, "DocumentNER")
    assert hasattr(ner, "extract_entities")
    assert hasattr(ner, "extract_custom_entities")
    assert hasattr(ner, "finalize_ner")
    assert hasattr(ner, "write_ner_json")


def test_ner_root_shim_private_symbols() -> None:
    """Private symbols are accessible via the root shim."""
    import ner  # noqa: F401

    # ``_SPACY_AVAILABLE`` is a module-level guard flag not declared in
    # ``__all__``; the sys.modules-replacement shim must still expose it.
    # ``_load_nlp`` and ``_entity_to_dict`` are private helpers; ``logger``
    # and the compiled regex patterns are likewise non-``__all__`` attributes.
    assert hasattr(ner, "_SPACY_AVAILABLE")
    assert hasattr(ner, "_SPACY_MODEL")
    assert hasattr(ner, "_load_nlp")
    assert hasattr(ner, "_entity_to_dict")
    assert hasattr(ner, "_CASE_NUMBER_PATTERN")
    assert hasattr(ner, "_BATES_NUMBER_PATTERN")
    assert hasattr(ner, "_EXHIBIT_REF_PATTERN")
    assert hasattr(ner, "logger")


def test_ner_not_in_feature_modules() -> None:
    """After Phase 2 migration, ner is not in the lazy-loader list."""
    from ocr_local.features import _FEATURE_MODULES

    assert "ner" not in _FEATURE_MODULES


def test_ner_not_in_compat_finder() -> None:
    """After Phase 2 migration, the compat finder no longer intercepts ner."""
    from ocr_local._compat_finder import _SUBPACKAGE_MODULES

    assert "ner" not in _SUBPACKAGE_MODULES["ocr_local.features"]


def test_ner_feature_attr_access() -> None:
    """ocr_local.features.ner is accessible as a package attribute."""
    import ocr_local.features as features
    import ocr_local.features.ner  # noqa: F401

    mod = features.ner
    assert mod.__name__ == "ocr_local.features.ner"


# ---------------------------------------------------------------------------
# preprocessing -- Phase 2 migration tests
# ---------------------------------------------------------------------------
def test_preprocessing_native_subpackage_import() -> None:
    """After Phase 2 migration, ocr_local.features.preprocessing is native."""
    mod = importlib.import_module("ocr_local.features.preprocessing")
    assert mod.__file__ is not None
    assert mod.__name__ == "ocr_local.features.preprocessing"


def test_preprocessing_root_shim_still_works() -> None:
    """Root-level 'import preprocessing' continues to work via the shim."""
    import preprocessing  # noqa: F401

    assert hasattr(preprocessing, "deskew_image")
    assert hasattr(preprocessing, "binarize_adaptive")
    assert hasattr(preprocessing, "denoise_bilateral")
    assert hasattr(preprocessing, "denoise_image")
    assert hasattr(preprocessing, "binarize_image")
    assert hasattr(preprocessing, "enhance_contrast")
    assert hasattr(preprocessing, "preprocess_for_ocr")
    assert hasattr(preprocessing, "denoise_nafnet")
    assert hasattr(preprocessing, "binarize_unet")


def test_preprocessing_root_shim_private_symbols() -> None:
    """Private symbols are accessible via the root shim."""
    import preprocessing  # noqa: F401

    # Module-level guards and caches are not in ``__all__``; the
    # sys.modules-replacement shim must still expose them for tests and
    # callers that peek at internals.
    assert hasattr(preprocessing, "_CV2_AVAILABLE")
    assert hasattr(preprocessing, "_ONNX_AVAILABLE")
    assert hasattr(preprocessing, "_pil_to_gray")
    assert hasattr(preprocessing, "_pil_to_bgr")
    assert hasattr(preprocessing, "_bgr_to_pil")
    assert hasattr(preprocessing, "_too_small")
    assert hasattr(preprocessing, "_load_onnx_model")
    assert hasattr(preprocessing, "_tile_inference")
    assert hasattr(preprocessing, "_onnx_model_cache")
    assert hasattr(preprocessing, "_onnx_cache_lock")
    assert hasattr(preprocessing, "_TILE_SIZE")
    assert hasattr(preprocessing, "logger")


def test_preprocessing_not_in_feature_modules() -> None:
    """After Phase 2 migration, preprocessing is not in the lazy-loader list."""
    from ocr_local.features import _FEATURE_MODULES

    assert "preprocessing" not in _FEATURE_MODULES


def test_preprocessing_not_in_compat_finder() -> None:
    """After Phase 2 migration, the compat finder no longer intercepts preprocessing."""
    from ocr_local._compat_finder import _SUBPACKAGE_MODULES

    assert "preprocessing" not in _SUBPACKAGE_MODULES["ocr_local.features"]


def test_preprocessing_feature_attr_access() -> None:
    """ocr_local.features.preprocessing is accessible as a package attribute."""
    import ocr_local.features.preprocessing  # noqa: F401
    from ocr_local import features

    mod = features.preprocessing
    assert mod.__name__ == "ocr_local.features.preprocessing"


# ---------------------------------------------------------------------------
# classification -- Phase 2 migration tests
# ---------------------------------------------------------------------------
def test_classification_native_subpackage_import() -> None:
    """ocr_local.features.classification imports natively from its physical path."""
    mod = importlib.import_module("ocr_local.features.classification")
    assert mod.__file__ is not None
    assert mod.__name__ == "ocr_local.features.classification"


def test_classification_root_shim_still_works() -> None:
    """Root-level 'import classification' continues to work via the shim."""
    import classification  # noqa: F401

    assert hasattr(classification, "DOCUMENT_TYPES")
    assert hasattr(classification, "DocumentClassification")
    assert hasattr(classification, "PageClassification")
    assert hasattr(classification, "classify_page_by_text")
    assert hasattr(classification, "classify_page_ensemble")
    assert hasattr(classification, "finalize_classification")
    assert hasattr(classification, "write_classification_json")


def test_classification_root_shim_private_symbols() -> None:
    """Private symbols are accessible via the root shim."""
    import classification  # noqa: F401

    # Module-level rules, caches, and weights are not in ``__all__``; the
    # sys.modules-replacement shim must still expose them for tests and
    # callers that peek at internals.
    assert hasattr(classification, "_TEXT_RULES")
    assert hasattr(classification, "_COMPILED_RULES")
    assert hasattr(classification, "_TEXT_WEIGHT")
    assert hasattr(classification, "_LAYOUT_WEIGHT")
    assert hasattr(classification, "_ML_WEIGHT")
    assert hasattr(classification, "_HEURISTIC_WEIGHT")
    assert hasattr(classification, "_ML_TO_BASE_TYPE")
    assert hasattr(classification, "_MONEY_PATTERN")
    assert hasattr(classification, "_CLASSIFICATION_PROFILES")
    assert hasattr(classification, "_ml_classifier_instance")
    assert hasattr(classification, "_ml_classifier_lock")
    assert hasattr(classification, "logger")


def test_classification_not_in_feature_modules() -> None:
    """After Phase 2 migration, classification is not in the lazy-loader list."""
    from ocr_local.features import _FEATURE_MODULES

    assert "classification" not in _FEATURE_MODULES


def test_classification_not_in_compat_finder() -> None:
    """After Phase 2 migration, the compat finder no longer intercepts classification."""
    from ocr_local._compat_finder import _SUBPACKAGE_MODULES

    assert "classification" not in _SUBPACKAGE_MODULES["ocr_local.features"]


def test_classification_feature_attr_access() -> None:
    """ocr_local.features.classification is accessible as a package attribute."""
    import ocr_local.features.classification  # noqa: F401
    from ocr_local import features

    mod = features.classification
    assert mod.__name__ == "ocr_local.features.classification"


# ---------------------------------------------------------------------------
# extraction -- Phase 2 migration tests
# ---------------------------------------------------------------------------
def test_extraction_native_subpackage_import() -> None:
    mod = importlib.import_module("ocr_local.features.extraction")
    assert mod.__file__ is not None
    assert mod.__name__ == "ocr_local.features.extraction"


def test_extraction_root_shim_still_works() -> None:
    import extraction  # noqa: F401

    assert hasattr(extraction, "DocumentExtraction")


def test_extraction_root_shim_private_symbols() -> None:
    import extraction  # noqa: F401

    assert hasattr(extraction, "_UIE_TYPE_MAP")
    assert hasattr(extraction, "_DATE_ISO")


def test_extraction_not_in_feature_modules() -> None:
    from ocr_local.features import _FEATURE_MODULES

    assert "extraction" not in _FEATURE_MODULES


def test_extraction_not_in_compat_finder() -> None:
    from ocr_local._compat_finder import _SUBPACKAGE_MODULES

    assert "extraction" not in _SUBPACKAGE_MODULES["ocr_local.features"]


def test_extraction_feature_attr_access() -> None:
    import ocr_local.features.extraction  # noqa: F401
    from ocr_local import features

    mod = features.extraction
    assert mod.__name__ == "ocr_local.features.extraction"


# ---------------------------------------------------------------------------
# custody -- Phase 2 migration tests
# ---------------------------------------------------------------------------
def test_custody_native_subpackage_import() -> None:
    mod = importlib.import_module("ocr_local.features.custody")
    assert mod.__file__ is not None
    assert mod.__name__ == "ocr_local.features.custody"


def test_custody_root_shim_still_works() -> None:
    import custody  # noqa: F401

    assert hasattr(custody, "CustodyChain")
    assert hasattr(custody, "compute_file_hash")
    assert hasattr(custody, "verify_custody_file")


def test_custody_root_shim_private_symbols() -> None:
    import custody  # noqa: F401

    # Module-level private/internal attributes
    assert hasattr(custody, "MAX_CUSTODY_RETRIES")
    assert hasattr(custody, "CUSTODY_RETRY_DELAYS")
    assert hasattr(custody, "EVENT_TYPES")
    assert hasattr(custody, "logger")


def test_custody_not_in_feature_modules() -> None:
    from ocr_local.features import _FEATURE_MODULES

    assert "custody" not in _FEATURE_MODULES


def test_custody_not_in_compat_finder() -> None:
    from ocr_local._compat_finder import _SUBPACKAGE_MODULES

    assert "custody" not in _SUBPACKAGE_MODULES["ocr_local.features"]


def test_custody_feature_attr_access() -> None:
    import ocr_local.features.custody  # noqa: F401
    from ocr_local import features

    mod = features.custody
    assert mod.__name__ == "ocr_local.features.custody"


# ---------------------------------------------------------------------------
# table_fallback -- Phase 2 migration tests
# ---------------------------------------------------------------------------
def test_table_fallback_native_subpackage_import() -> None:
    mod = importlib.import_module("ocr_local.features.table_fallback")
    assert mod.__file__ is not None
    assert mod.__name__ == "ocr_local.features.table_fallback"


def test_table_fallback_root_shim_still_works() -> None:
    import table_fallback  # noqa: F401

    assert hasattr(table_fallback, "crop_table_region")
    assert hasattr(table_fallback, "TableRegion")
    assert hasattr(table_fallback, "ENABLE_TABLE_FALLBACK")


def test_table_fallback_root_shim_private_symbols() -> None:
    import table_fallback  # noqa: F401

    # Module-level private constants/helpers and the logger are not in
    # ``__all__``; the sys.modules-replacement shim must still expose them.
    assert hasattr(table_fallback, "_CONFIDENCE_CLOSE_DELTA")
    assert hasattr(table_fallback, "_CV2_AVAILABLE")
    assert hasattr(table_fallback, "_preprocess_enhance")
    assert hasattr(table_fallback, "logger")


def test_table_fallback_not_in_feature_modules() -> None:
    from ocr_local.features import _FEATURE_MODULES

    assert "table_fallback" not in _FEATURE_MODULES


def test_table_fallback_not_in_compat_finder() -> None:
    from ocr_local._compat_finder import _SUBPACKAGE_MODULES

    assert "table_fallback" not in _SUBPACKAGE_MODULES["ocr_local.features"]


def test_table_fallback_feature_attr_access() -> None:
    import ocr_local.features.table_fallback  # noqa: F401
    from ocr_local import features

    mod = features.table_fallback
    assert mod.__name__ == "ocr_local.features.table_fallback"


# ---------------------------------------------------------------------------
# signature_verification -- Phase 2 migration tests
# ---------------------------------------------------------------------------
def test_signature_verification_native_subpackage_import() -> None:
    mod = importlib.import_module("ocr_local.features.signature_verification")
    assert mod.__file__ is not None
    assert mod.__name__ == "ocr_local.features.signature_verification"


def test_signature_verification_root_shim_still_works() -> None:
    import signature_verification  # noqa: F401

    assert hasattr(signature_verification, "SignatureCandidate")
    assert hasattr(signature_verification, "PageSignatureVerification")
    assert hasattr(signature_verification, "DocumentSignatureVerification")
    assert hasattr(signature_verification, "analyze_signature_page")
    assert hasattr(signature_verification, "finalize_signature_verification")
    assert hasattr(signature_verification, "write_signature_verification_json")


def test_signature_verification_root_shim_private_symbols() -> None:
    import signature_verification  # noqa: F401

    # Module-level private constants/helpers and the logger are not in
    # ``__all__``; the sys.modules-replacement shim must still expose them.
    assert hasattr(signature_verification, "_SIGNATURE_KEYWORDS")
    assert hasattr(signature_verification, "_SIGNATURE_LINE_MARKERS")
    assert hasattr(signature_verification, "_PRESENCE_MIN_INK_RATIO")
    assert hasattr(signature_verification, "_to_grayscale_array")
    assert hasattr(signature_verification, "_stroke_complexity")
    assert hasattr(signature_verification, "logger")


def test_signature_verification_from_import() -> None:
    from signature_verification import (
        SignatureCandidate,
        analyze_signature_page,
        finalize_signature_verification,
    )

    assert SignatureCandidate is not None
    assert callable(analyze_signature_page)
    assert callable(finalize_signature_verification)


def test_signature_verification_root_and_native_are_same_object() -> None:
    import signature_verification

    native = importlib.import_module("ocr_local.features.signature_verification")
    assert signature_verification is native


def test_signature_verification_not_in_feature_modules() -> None:
    from ocr_local.features import _FEATURE_MODULES

    assert "signature_verification" not in _FEATURE_MODULES


def test_signature_verification_not_in_compat_finder() -> None:
    from ocr_local._compat_finder import _SUBPACKAGE_MODULES

    assert "signature_verification" not in _SUBPACKAGE_MODULES["ocr_local.features"]


def test_signature_verification_feature_attr_access() -> None:
    import ocr_local.features.signature_verification  # noqa: F401
    from ocr_local import features

    mod = features.signature_verification
    assert mod.__name__ == "ocr_local.features.signature_verification"


# ---------------------------------------------------------------------------
# cost_tracking -- Phase 2 migration tests
# ---------------------------------------------------------------------------
def test_cost_tracking_native_subpackage_import() -> None:
    mod = importlib.import_module("ocr_local.features.cost_tracking")
    assert mod.__file__ is not None
    assert mod.__name__ == "ocr_local.features.cost_tracking"


def test_cost_tracking_root_shim_still_works() -> None:
    import cost_tracking  # noqa: F401

    assert hasattr(cost_tracking, "BillingFormula")
    assert hasattr(cost_tracking, "TenantUsage")
    assert hasattr(cost_tracking, "CostTracker")
    assert hasattr(cost_tracking, "get_billing_formula")
    assert hasattr(cost_tracking, "validate_billing_formula")
    assert hasattr(cost_tracking, "get_tracker")
    assert hasattr(cost_tracking, "reset_global_tracker")
    assert hasattr(cost_tracking, "ENABLE_COST_TRACKING")
    assert hasattr(cost_tracking, "BILLING_FORMULA_VERSION")


def test_cost_tracking_root_shim_private_symbols() -> None:
    import cost_tracking  # noqa: F401

    # Module-level private constants and the logger are not in ``__all__``;
    # the sys.modules-replacement shim must still expose them for tests and
    # callers that peek at internals.
    assert hasattr(cost_tracking, "_SEMVER_RE")
    assert hasattr(cost_tracking, "logger")


def test_cost_tracking_from_import() -> None:
    from cost_tracking import (
        BillingFormula,
        CostTracker,
        get_billing_formula,
        validate_billing_formula,
    )

    assert BillingFormula is not None
    assert CostTracker is not None
    assert callable(get_billing_formula)
    assert callable(validate_billing_formula)


def test_cost_tracking_root_and_native_are_same_object() -> None:
    import cost_tracking

    native = importlib.import_module("ocr_local.features.cost_tracking")
    assert cost_tracking is native


def test_cost_tracking_not_in_feature_modules() -> None:
    from ocr_local.features import _FEATURE_MODULES

    assert "cost_tracking" not in _FEATURE_MODULES


def test_cost_tracking_not_in_compat_finder() -> None:
    from ocr_local._compat_finder import _SUBPACKAGE_MODULES

    assert "cost_tracking" not in _SUBPACKAGE_MODULES["ocr_local.features"]


def test_cost_tracking_feature_attr_access() -> None:
    import ocr_local.features.cost_tracking  # noqa: F401
    from ocr_local import features

    mod = features.cost_tracking
    assert mod.__name__ == "ocr_local.features.cost_tracking"


# ---------------------------------------------------------------------------
# vertical_text -- Phase 2 migration tests (final features migration)
# ---------------------------------------------------------------------------
def test_vertical_text_native_subpackage_import() -> None:
    mod = importlib.import_module("ocr_local.features.vertical_text")
    assert mod.__file__ is not None
    assert mod.__name__ == "ocr_local.features.vertical_text"


def test_vertical_text_root_shim_still_works() -> None:
    import vertical_text  # noqa: F401

    assert hasattr(vertical_text, "is_cjk_char")
    assert hasattr(vertical_text, "contains_cjk")
    assert hasattr(vertical_text, "is_vertical_text_box")
    assert hasattr(vertical_text, "classify_page_text_direction")
    assert hasattr(vertical_text, "VerticalTextAnalysis")
    assert hasattr(vertical_text, "DocumentVerticalText")
    assert hasattr(vertical_text, "analyze_page_vertical_text")
    assert hasattr(vertical_text, "finalize_vertical_text")
    assert hasattr(vertical_text, "write_vertical_analysis_json")
    assert hasattr(vertical_text, "ENABLE_VERTICAL_TEXT")
    assert hasattr(vertical_text, "CJK_UNICODE_RANGES")


def test_vertical_text_root_shim_private_symbols() -> None:
    import vertical_text  # noqa: F401

    # Module-level private constants/helpers and the logger are not in
    # ``__all__``; the sys.modules-replacement shim must still expose them.
    assert hasattr(vertical_text, "_PIL_AVAILABLE")
    assert hasattr(vertical_text, "_box_dimensions")
    assert hasattr(vertical_text, "_box_center_x")
    assert hasattr(vertical_text, "_box_top_y")
    assert hasattr(vertical_text, "logger")


def test_vertical_text_from_import() -> None:
    from vertical_text import (
        DocumentVerticalText,
        VerticalTextAnalysis,
        analyze_page_vertical_text,
        finalize_vertical_text,
    )

    assert DocumentVerticalText is not None
    assert VerticalTextAnalysis is not None
    assert callable(analyze_page_vertical_text)
    assert callable(finalize_vertical_text)


def test_vertical_text_root_and_native_are_same_object() -> None:
    import vertical_text

    native = importlib.import_module("ocr_local.features.vertical_text")
    assert vertical_text is native


def test_vertical_text_not_in_feature_modules() -> None:
    from ocr_local.features import _FEATURE_MODULES

    assert "vertical_text" not in _FEATURE_MODULES


def test_vertical_text_not_in_compat_finder() -> None:
    from ocr_local._compat_finder import _SUBPACKAGE_MODULES

    assert "vertical_text" not in _SUBPACKAGE_MODULES.get(
        "ocr_local.features", frozenset()
    )


def test_vertical_text_feature_attr_access() -> None:
    import ocr_local.features.vertical_text  # noqa: F401
    from ocr_local import features

    mod = features.vertical_text
    assert mod.__name__ == "ocr_local.features.vertical_text"


# ---------------------------------------------------------------------------
# Phase 2 -- ml/ migrations
# ---------------------------------------------------------------------------
# layoutlm_calibration -- Phase 2 migration tests
# ---------------------------------------------------------------------------


def test_layoutlm_calibration_native_subpackage_import() -> None:
    mod = importlib.import_module("ocr_local.ml.layoutlm_calibration")
    assert mod.__file__ is not None
    assert mod.__name__ == "ocr_local.ml.layoutlm_calibration"


def test_layoutlm_calibration_root_shim_still_works() -> None:
    import layoutlm_calibration  # noqa: F401

    assert hasattr(layoutlm_calibration, "CalibrationMethod")
    assert hasattr(layoutlm_calibration, "CalibrationConfig")
    assert hasattr(layoutlm_calibration, "ConfidenceCalibrator")
    assert hasattr(layoutlm_calibration, "calibrate_entity_confidence")
    assert hasattr(layoutlm_calibration, "compute_ece")
    assert hasattr(layoutlm_calibration, "get_default_calibrator")


def test_layoutlm_calibration_root_shim_private_symbols() -> None:
    import layoutlm_calibration  # noqa: F401

    # Private helpers must be exposed via the sys.modules-replacement shim
    assert hasattr(layoutlm_calibration, "_resolve_method")
    assert hasattr(layoutlm_calibration, "_softmax")
    assert hasattr(layoutlm_calibration, "_sigmoid")


def test_layoutlm_calibration_from_import() -> None:
    from layoutlm_calibration import (
        CalibrationConfig,
        CalibrationMethod,
        ConfidenceCalibrator,
    )

    assert isinstance(CalibrationConfig, type)
    assert isinstance(ConfidenceCalibrator, type)
    assert CalibrationMethod is not None


def test_layoutlm_calibration_root_and_native_are_same_object() -> None:
    import layoutlm_calibration

    native = importlib.import_module("ocr_local.ml.layoutlm_calibration")
    assert layoutlm_calibration is native


def test_layoutlm_calibration_not_in_ml_modules() -> None:
    from ocr_local.ml import _ML_MODULES

    assert "layoutlm_calibration" not in _ML_MODULES


def test_layoutlm_calibration_not_in_compat_finder() -> None:
    from ocr_local._compat_finder import _SUBPACKAGE_MODULES

    assert "layoutlm_calibration" not in _SUBPACKAGE_MODULES.get(
        "ocr_local.ml", frozenset()
    )


def test_layoutlm_calibration_ml_attr_access() -> None:
    import ocr_local.ml.layoutlm_calibration  # noqa: F401
    from ocr_local import ml

    mod = ml.layoutlm_calibration
    assert mod.__name__ == "ocr_local.ml.layoutlm_calibration"


# ---------------------------------------------------------------------------
# layoutlm_data -- Phase 2 migration tests
# ---------------------------------------------------------------------------


def test_layoutlm_data_native_subpackage_import() -> None:
    mod = importlib.import_module("ocr_local.ml.layoutlm_data")
    assert mod.__file__ is not None
    assert mod.__name__ == "ocr_local.ml.layoutlm_data"


def test_layoutlm_data_root_shim_still_works() -> None:
    import layoutlm_data  # noqa: F401

    assert hasattr(layoutlm_data, "AnnotatedWord")
    assert hasattr(layoutlm_data, "AnnotatedPage")
    assert hasattr(layoutlm_data, "load_custom_jsonl")
    assert hasattr(layoutlm_data, "create_hf_dataset")


def test_layoutlm_data_root_shim_private_symbols() -> None:
    import layoutlm_data  # noqa: F401

    # Module-level non-``__all__`` attributes must be exposed via the shim
    assert hasattr(layoutlm_data, "logger")


def test_layoutlm_data_from_import() -> None:
    from layoutlm_data import AnnotatedPage, AnnotatedWord, load_custom_jsonl

    assert isinstance(AnnotatedWord, type)
    assert isinstance(AnnotatedPage, type)
    assert callable(load_custom_jsonl)


def test_layoutlm_data_root_and_native_are_same_object() -> None:
    import layoutlm_data

    native = importlib.import_module("ocr_local.ml.layoutlm_data")
    assert layoutlm_data is native


def test_layoutlm_data_not_in_ml_modules() -> None:
    from ocr_local.ml import _ML_MODULES

    assert "layoutlm_data" not in _ML_MODULES


def test_layoutlm_data_not_in_compat_finder() -> None:
    from ocr_local._compat_finder import _SUBPACKAGE_MODULES

    assert "layoutlm_data" not in _SUBPACKAGE_MODULES.get(
        "ocr_local.ml", frozenset()
    )


def test_layoutlm_data_ml_attr_access() -> None:
    import ocr_local.ml.layoutlm_data  # noqa: F401
    from ocr_local import ml

    mod = ml.layoutlm_data
    assert mod.__name__ == "ocr_local.ml.layoutlm_data"


# ---------------------------------------------------------------------------
# layoutlm_evaluate -- Phase 2 migration tests
# ---------------------------------------------------------------------------


def test_layoutlm_evaluate_native_subpackage_import() -> None:
    mod = importlib.import_module("ocr_local.ml.layoutlm_evaluate")
    assert mod.__file__ is not None
    assert mod.__name__ == "ocr_local.ml.layoutlm_evaluate"


def test_layoutlm_evaluate_root_shim_still_works() -> None:
    import layoutlm_evaluate  # noqa: F401

    assert hasattr(layoutlm_evaluate, "evaluate_model")
    assert hasattr(layoutlm_evaluate, "evaluate_predictions")


def test_layoutlm_evaluate_root_shim_private_symbols() -> None:
    import layoutlm_evaluate  # noqa: F401

    # Private helpers must be exposed via the sys.modules-replacement shim
    assert hasattr(layoutlm_evaluate, "_empty_report")
    assert hasattr(layoutlm_evaluate, "_write_report")
    assert hasattr(layoutlm_evaluate, "logger")


def test_layoutlm_evaluate_from_import() -> None:
    from layoutlm_evaluate import evaluate_model, evaluate_predictions

    assert callable(evaluate_model)
    assert callable(evaluate_predictions)


def test_layoutlm_evaluate_root_and_native_are_same_object() -> None:
    import layoutlm_evaluate

    native = importlib.import_module("ocr_local.ml.layoutlm_evaluate")
    assert layoutlm_evaluate is native


def test_layoutlm_evaluate_not_in_ml_modules() -> None:
    from ocr_local.ml import _ML_MODULES

    assert "layoutlm_evaluate" not in _ML_MODULES


def test_layoutlm_evaluate_not_in_compat_finder() -> None:
    from ocr_local._compat_finder import _SUBPACKAGE_MODULES

    assert "layoutlm_evaluate" not in _SUBPACKAGE_MODULES.get(
        "ocr_local.ml", frozenset()
    )


def test_layoutlm_evaluate_ml_attr_access() -> None:
    import ocr_local.ml.layoutlm_evaluate  # noqa: F401
    from ocr_local import ml

    mod = ml.layoutlm_evaluate
    assert mod.__name__ == "ocr_local.ml.layoutlm_evaluate"


# ---------------------------------------------------------------------------
# layoutlm_finetune -- Phase 2 migration tests
# ---------------------------------------------------------------------------


def test_layoutlm_finetune_native_subpackage_import() -> None:
    mod = importlib.import_module("ocr_local.ml.layoutlm_finetune")
    assert mod.__file__ is not None
    assert mod.__name__ == "ocr_local.ml.layoutlm_finetune"


def test_layoutlm_finetune_root_shim_still_works() -> None:
    import layoutlm_finetune  # noqa: F401

    assert hasattr(layoutlm_finetune, "FineTuneConfig")
    assert hasattr(layoutlm_finetune, "run_finetuning")
    assert hasattr(layoutlm_finetune, "main")
    assert hasattr(layoutlm_finetune, "DEFAULT_BASE_MODEL")


def test_layoutlm_finetune_root_shim_private_symbols() -> None:
    import layoutlm_finetune  # noqa: F401

    # Private helpers must be exposed via the sys.modules-replacement shim
    assert hasattr(layoutlm_finetune, "_build_compute_metrics")
    assert hasattr(layoutlm_finetune, "_parse_args")
    assert hasattr(layoutlm_finetune, "logger")


def test_layoutlm_finetune_from_import() -> None:
    from layoutlm_finetune import FineTuneConfig, run_finetuning

    assert isinstance(FineTuneConfig, type)
    assert callable(run_finetuning)


def test_layoutlm_finetune_root_and_native_are_same_object() -> None:
    import layoutlm_finetune

    native = importlib.import_module("ocr_local.ml.layoutlm_finetune")
    assert layoutlm_finetune is native


def test_layoutlm_finetune_not_in_ml_modules() -> None:
    from ocr_local.ml import _ML_MODULES

    assert "layoutlm_finetune" not in _ML_MODULES


def test_layoutlm_finetune_not_in_compat_finder() -> None:
    from ocr_local._compat_finder import _SUBPACKAGE_MODULES

    assert "layoutlm_finetune" not in _SUBPACKAGE_MODULES.get(
        "ocr_local.ml", frozenset()
    )


def test_layoutlm_finetune_ml_attr_access() -> None:
    import ocr_local.ml.layoutlm_finetune  # noqa: F401
    from ocr_local import ml

    mod = ml.layoutlm_finetune
    assert mod.__name__ == "ocr_local.ml.layoutlm_finetune"


# ---------------------------------------------------------------------------
# layoutlm_labels -- Phase 2 migration tests
# ---------------------------------------------------------------------------


def test_layoutlm_labels_native_subpackage_import() -> None:
    mod = importlib.import_module("ocr_local.ml.layoutlm_labels")
    assert mod.__file__ is not None
    assert mod.__name__ == "ocr_local.ml.layoutlm_labels"


def test_layoutlm_labels_root_shim_still_works() -> None:
    import layoutlm_labels  # noqa: F401

    assert hasattr(layoutlm_labels, "LabelSet")
    assert hasattr(layoutlm_labels, "expand_to_bio")
    assert hasattr(layoutlm_labels, "build_label_set")
    assert hasattr(layoutlm_labels, "load_label_set")
    assert hasattr(layoutlm_labels, "get_active_label_set")
    assert hasattr(layoutlm_labels, "DEFAULT_TYPE_MAP")
    assert hasattr(layoutlm_labels, "BUILTIN_LABEL_SETS")


def test_layoutlm_labels_root_shim_private_symbols() -> None:
    import layoutlm_labels  # noqa: F401

    # Module-level non-``__all__`` attributes must be exposed via the shim
    assert hasattr(layoutlm_labels, "logger")


def test_layoutlm_labels_from_import() -> None:
    from layoutlm_labels import LabelSet, build_label_set, expand_to_bio

    assert isinstance(LabelSet, type)
    assert callable(expand_to_bio)
    assert callable(build_label_set)


def test_layoutlm_labels_root_and_native_are_same_object() -> None:
    import layoutlm_labels

    native = importlib.import_module("ocr_local.ml.layoutlm_labels")
    assert layoutlm_labels is native


def test_layoutlm_labels_not_in_ml_modules() -> None:
    from ocr_local.ml import _ML_MODULES

    assert "layoutlm_labels" not in _ML_MODULES


def test_layoutlm_labels_not_in_compat_finder() -> None:
    from ocr_local._compat_finder import _SUBPACKAGE_MODULES

    assert "layoutlm_labels" not in _SUBPACKAGE_MODULES.get(
        "ocr_local.ml", frozenset()
    )


def test_layoutlm_labels_ml_attr_access() -> None:
    import ocr_local.ml.layoutlm_labels  # noqa: F401
    from ocr_local import ml

    mod = ml.layoutlm_labels
    assert mod.__name__ == "ocr_local.ml.layoutlm_labels"


# ---------------------------------------------------------------------------
# layoutlm_model_registry -- Phase 2 migration tests
# ---------------------------------------------------------------------------


def test_layoutlm_model_registry_native_subpackage_import() -> None:
    mod = importlib.import_module("ocr_local.ml.layoutlm_model_registry")
    assert mod.__file__ is not None
    assert mod.__name__ == "ocr_local.ml.layoutlm_model_registry"


def test_layoutlm_model_registry_root_shim_still_works() -> None:
    import layoutlm_model_registry  # noqa: F401

    assert hasattr(layoutlm_model_registry, "ModelRegistry")
    assert hasattr(layoutlm_model_registry, "ModelRegistryEntry")
    assert hasattr(layoutlm_model_registry, "ResolvedModelSelection")
    assert hasattr(layoutlm_model_registry, "resolve_active_model_selection")
    assert hasattr(layoutlm_model_registry, "DEFAULT_REGISTRY_DIR")
    assert hasattr(layoutlm_model_registry, "MANIFEST_FILENAME")


def test_layoutlm_model_registry_root_shim_private_symbols() -> None:
    import layoutlm_model_registry  # noqa: F401

    # Module-level non-``__all__`` attributes must be exposed via the shim
    assert hasattr(layoutlm_model_registry, "logger")


def test_layoutlm_model_registry_from_import() -> None:
    from layoutlm_model_registry import (
        ModelRegistry,
        ModelRegistryEntry,
        ResolvedModelSelection,
    )

    assert isinstance(ModelRegistry, type)
    assert isinstance(ModelRegistryEntry, type)
    assert isinstance(ResolvedModelSelection, type)


def test_layoutlm_model_registry_root_and_native_are_same_object() -> None:
    import layoutlm_model_registry

    native = importlib.import_module("ocr_local.ml.layoutlm_model_registry")
    assert layoutlm_model_registry is native


def test_layoutlm_model_registry_not_in_ml_modules() -> None:
    from ocr_local.ml import _ML_MODULES

    assert "layoutlm_model_registry" not in _ML_MODULES


def test_layoutlm_model_registry_not_in_compat_finder() -> None:
    from ocr_local._compat_finder import _SUBPACKAGE_MODULES

    assert "layoutlm_model_registry" not in _SUBPACKAGE_MODULES.get(
        "ocr_local.ml", frozenset()
    )


def test_layoutlm_model_registry_ml_attr_access() -> None:
    import ocr_local.ml.layoutlm_model_registry  # noqa: F401
    from ocr_local import ml

    mod = ml.layoutlm_model_registry
    assert mod.__name__ == "ocr_local.ml.layoutlm_model_registry"


# ---------------------------------------------------------------------------
# layoutlm_summarization -- Phase 2 migration tests (final ml migration)
# ---------------------------------------------------------------------------


def test_layoutlm_summarization_native_subpackage_import() -> None:
    mod = importlib.import_module("ocr_local.ml.layoutlm_summarization")
    assert mod.__file__ is not None
    assert mod.__name__ == "ocr_local.ml.layoutlm_summarization"


def test_layoutlm_summarization_root_shim_still_works() -> None:
    import layoutlm_summarization  # noqa: F401

    assert hasattr(layoutlm_summarization, "SummarizationMethod")
    assert hasattr(layoutlm_summarization, "SummarizationConfig")
    assert hasattr(layoutlm_summarization, "SummarySentence")
    assert hasattr(layoutlm_summarization, "DocumentSummary")
    assert hasattr(layoutlm_summarization, "summarize_document")
    assert hasattr(layoutlm_summarization, "summarize_from_files")
    assert hasattr(layoutlm_summarization, "summary_to_dict")


def test_layoutlm_summarization_root_shim_private_symbols() -> None:
    import layoutlm_summarization  # noqa: F401

    # Module-level private constants and helpers must be exposed via the shim
    assert hasattr(layoutlm_summarization, "_DEFAULT_METHOD")
    assert hasattr(layoutlm_summarization, "_DEFAULT_MAX_SENTENCES")
    assert hasattr(layoutlm_summarization, "_METHOD_LOOKUP")
    assert hasattr(layoutlm_summarization, "_default_config")
    assert hasattr(layoutlm_summarization, "_split_sentences")
    assert hasattr(layoutlm_summarization, "_textrank_scores")
    assert hasattr(layoutlm_summarization, "logger")


def test_layoutlm_summarization_from_import() -> None:
    from layoutlm_summarization import (
        DocumentSummary,
        SummarizationConfig,
        SummarizationMethod,
        summarize_document,
    )

    assert isinstance(SummarizationConfig, type)
    assert isinstance(DocumentSummary, type)
    assert SummarizationMethod is not None
    assert callable(summarize_document)


def test_layoutlm_summarization_root_and_native_are_same_object() -> None:
    import layoutlm_summarization

    native = importlib.import_module("ocr_local.ml.layoutlm_summarization")
    assert layoutlm_summarization is native


def test_layoutlm_summarization_not_in_ml_modules() -> None:
    from ocr_local.ml import _ML_MODULES

    assert "layoutlm_summarization" not in _ML_MODULES


def test_layoutlm_summarization_not_in_compat_finder() -> None:
    from ocr_local._compat_finder import _SUBPACKAGE_MODULES

    assert "layoutlm_summarization" not in _SUBPACKAGE_MODULES.get(
        "ocr_local.ml", frozenset()
    )


def test_layoutlm_summarization_ml_attr_access() -> None:
    import ocr_local.ml.layoutlm_summarization  # noqa: F401
    from ocr_local import ml

    mod = ml.layoutlm_summarization
    assert mod.__name__ == "ocr_local.ml.layoutlm_summarization"


# ---------------------------------------------------------------------------
# adaptive_batch -- Phase 2 migration tests (infra)
# ---------------------------------------------------------------------------


def test_adaptive_batch_native_subpackage_import() -> None:
    mod = importlib.import_module("ocr_local.infra.adaptive_batch")
    assert mod.__file__ is not None
    assert mod.__name__ == "ocr_local.infra.adaptive_batch"


def test_adaptive_batch_root_shim_still_works() -> None:
    import adaptive_batch  # noqa: F401

    assert hasattr(adaptive_batch, "BatchStrategy")
    assert hasattr(adaptive_batch, "BatchConfig")
    assert hasattr(adaptive_batch, "PageComplexity")
    assert hasattr(adaptive_batch, "BatchResult")
    assert hasattr(adaptive_batch, "AdaptiveBatchSizer")


def test_adaptive_batch_root_shim_private_symbols() -> None:
    import adaptive_batch  # noqa: F401

    # Module-level private constants must be exposed via the shim
    assert hasattr(adaptive_batch, "_AREA_WEIGHT")
    assert hasattr(adaptive_batch, "_FILE_SIZE_WEIGHT")
    assert hasattr(adaptive_batch, "_TABLE_WEIGHT")
    assert hasattr(adaptive_batch, "_IMAGE_WEIGHT")
    assert hasattr(adaptive_batch, "_REF_AREA")
    assert hasattr(adaptive_batch, "_REF_FILE_SIZE")
    assert hasattr(adaptive_batch, "logger")


def test_adaptive_batch_from_import() -> None:
    from adaptive_batch import (
        AdaptiveBatchSizer,
        BatchConfig,
        BatchStrategy,
        PageComplexity,
    )

    assert isinstance(BatchConfig, type)
    assert isinstance(AdaptiveBatchSizer, type)
    assert isinstance(PageComplexity, type)
    assert BatchStrategy is not None


def test_adaptive_batch_root_and_native_are_same_object() -> None:
    import adaptive_batch

    native = importlib.import_module("ocr_local.infra.adaptive_batch")
    assert adaptive_batch is native


def test_adaptive_batch_not_in_infra_modules() -> None:
    from ocr_local.infra import _INFRA_MODULES

    assert "adaptive_batch" not in _INFRA_MODULES


def test_adaptive_batch_not_in_compat_finder() -> None:
    from ocr_local._compat_finder import _SUBPACKAGE_MODULES

    assert "adaptive_batch" not in _SUBPACKAGE_MODULES.get(
        "ocr_local.infra", frozenset()
    )


def test_adaptive_batch_infra_attr_access() -> None:
    import ocr_local.infra.adaptive_batch  # noqa: F401
    from ocr_local import infra

    mod = infra.adaptive_batch
    assert mod.__name__ == "ocr_local.infra.adaptive_batch"


# ---------------------------------------------------------------------------
# benchmark_ocr -- Phase 2 migration tests (infra)
# ---------------------------------------------------------------------------


def test_benchmark_ocr_native_subpackage_import() -> None:
    mod = importlib.import_module("ocr_local.infra.benchmark_ocr")
    assert mod.__file__ is not None
    assert mod.__name__ == "ocr_local.infra.benchmark_ocr"


def test_benchmark_ocr_root_shim_still_works() -> None:
    import benchmark_ocr  # noqa: F401

    assert hasattr(benchmark_ocr, "generate_test_page")
    assert hasattr(benchmark_ocr, "benchmark_paddle")
    assert hasattr(benchmark_ocr, "benchmark_tesseract")
    assert hasattr(benchmark_ocr, "format_comparison_table")
    assert hasattr(benchmark_ocr, "run_benchmarks")


def test_benchmark_ocr_root_shim_private_symbols() -> None:
    import benchmark_ocr  # noqa: F401

    # Module-level private constants and helpers must be exposed via the shim
    assert hasattr(benchmark_ocr, "_LANGUAGE_CHARSETS")
    assert hasattr(benchmark_ocr, "_overlay_language_text")
    assert hasattr(benchmark_ocr, "_get_process_memory_mb")
    assert hasattr(benchmark_ocr, "_compile_results")
    assert hasattr(benchmark_ocr, "_detect_available_backends")
    assert hasattr(benchmark_ocr, "_load_document_pages")
    assert hasattr(benchmark_ocr, "logger")


def test_benchmark_ocr_from_import() -> None:
    from benchmark_ocr import (
        DEFAULT_PAGES,
        generate_test_page,
        run_benchmarks,
    )

    assert callable(generate_test_page)
    assert callable(run_benchmarks)
    assert isinstance(DEFAULT_PAGES, int)


def test_benchmark_ocr_root_and_native_are_same_object() -> None:
    import benchmark_ocr

    native = importlib.import_module("ocr_local.infra.benchmark_ocr")
    assert benchmark_ocr is native


def test_benchmark_ocr_not_in_infra_modules() -> None:
    from ocr_local.infra import _INFRA_MODULES

    assert "benchmark_ocr" not in _INFRA_MODULES


def test_benchmark_ocr_not_in_compat_finder() -> None:
    from ocr_local._compat_finder import _SUBPACKAGE_MODULES

    assert "benchmark_ocr" not in _SUBPACKAGE_MODULES.get(
        "ocr_local.infra", frozenset()
    )


def test_benchmark_ocr_infra_attr_access() -> None:
    import ocr_local.infra.benchmark_ocr  # noqa: F401
    from ocr_local import infra

    mod = infra.benchmark_ocr
    assert mod.__name__ == "ocr_local.infra.benchmark_ocr"


# ---------------------------------------------------------------------------
# benchmark_pipeline -- Phase 2 migration tests (infra)
# ---------------------------------------------------------------------------


def test_benchmark_pipeline_native_subpackage_import() -> None:
    mod = importlib.import_module("ocr_local.infra.benchmark_pipeline")
    assert mod.__file__ is not None
    assert mod.__name__ == "ocr_local.infra.benchmark_pipeline"


def test_benchmark_pipeline_root_shim_still_works() -> None:
    import benchmark_pipeline  # noqa: F401

    assert hasattr(benchmark_pipeline, "BenchmarkMetrics")
    assert hasattr(benchmark_pipeline, "IOCounters")
    assert hasattr(benchmark_pipeline, "InstrumentedQueue")
    assert hasattr(benchmark_pipeline, "simulate_pipeline")
    assert hasattr(benchmark_pipeline, "instrument_live_pipeline")
    assert hasattr(benchmark_pipeline, "run_benchmark")


def test_benchmark_pipeline_root_shim_private_symbols() -> None:
    import benchmark_pipeline  # noqa: F401

    # Module-level private helpers must be exposed via the shim
    assert hasattr(benchmark_pipeline, "_sample_ms")
    assert hasattr(benchmark_pipeline, "_percentile")
    assert hasattr(benchmark_pipeline, "_memory_sampler")


def test_benchmark_pipeline_from_import() -> None:
    from benchmark_pipeline import (
        BenchmarkMetrics,
        IOCounters,
        simulate_pipeline,
    )

    assert isinstance(BenchmarkMetrics, type)
    assert isinstance(IOCounters, type)
    assert callable(simulate_pipeline)


def test_benchmark_pipeline_root_and_native_are_same_object() -> None:
    import benchmark_pipeline

    native = importlib.import_module("ocr_local.infra.benchmark_pipeline")
    assert benchmark_pipeline is native


def test_benchmark_pipeline_not_in_infra_modules() -> None:
    from ocr_local.infra import _INFRA_MODULES

    assert "benchmark_pipeline" not in _INFRA_MODULES


def test_benchmark_pipeline_not_in_compat_finder() -> None:
    from ocr_local._compat_finder import _SUBPACKAGE_MODULES

    assert "benchmark_pipeline" not in _SUBPACKAGE_MODULES.get(
        "ocr_local.infra", frozenset()
    )


def test_benchmark_pipeline_infra_attr_access() -> None:
    import ocr_local.infra.benchmark_pipeline  # noqa: F401
    from ocr_local import infra

    mod = infra.benchmark_pipeline
    assert mod.__name__ == "ocr_local.infra.benchmark_pipeline"


# ---------------------------------------------------------------------------
# engine_selection -- Phase 2 migration tests (infra)
# ---------------------------------------------------------------------------


def test_engine_selection_native_subpackage_import() -> None:
    mod = importlib.import_module("ocr_local.infra.engine_selection")
    assert mod.__file__ is not None
    assert mod.__name__ == "ocr_local.infra.engine_selection"


def test_engine_selection_root_shim_still_works() -> None:
    import engine_selection  # noqa: F401

    assert hasattr(engine_selection, "DocumentQuality")
    assert hasattr(engine_selection, "analyze_page_quality")
    assert hasattr(engine_selection, "select_engine")
    assert hasattr(engine_selection, "select_engine_for_page")
    assert hasattr(engine_selection, "ENGINE_SELECTION")


def test_engine_selection_root_shim_private_symbols() -> None:
    import engine_selection  # noqa: F401

    # Module-level private helpers must be exposed via the shim
    assert hasattr(engine_selection, "_estimate_skew")
    assert hasattr(engine_selection, "_is_easyocr_available")
    assert hasattr(engine_selection, "logger")


def test_engine_selection_from_import() -> None:
    from engine_selection import (
        DocumentQuality,
        analyze_page_quality,
        select_engine,
    )

    assert isinstance(DocumentQuality, type)
    assert callable(analyze_page_quality)
    assert callable(select_engine)


def test_engine_selection_root_and_native_are_same_object() -> None:
    import engine_selection

    native = importlib.import_module("ocr_local.infra.engine_selection")
    assert engine_selection is native


def test_engine_selection_not_in_infra_modules() -> None:
    from ocr_local.infra import _INFRA_MODULES

    assert "engine_selection" not in _INFRA_MODULES


def test_engine_selection_not_in_compat_finder() -> None:
    from ocr_local._compat_finder import _SUBPACKAGE_MODULES

    assert "engine_selection" not in _SUBPACKAGE_MODULES.get(
        "ocr_local.infra", frozenset()
    )


def test_engine_selection_infra_attr_access() -> None:
    import ocr_local.infra.engine_selection  # noqa: F401
    from ocr_local import infra

    mod = infra.engine_selection
    assert mod.__name__ == "ocr_local.infra.engine_selection"


# ---------------------------------------------------------------------------
# gpu_optimization -- Phase 2 migration tests (infra)
# ---------------------------------------------------------------------------


def test_gpu_optimization_native_subpackage_import() -> None:
    mod = importlib.import_module("ocr_local.infra.gpu_optimization")
    assert mod.__file__ is not None
    assert mod.__name__ == "ocr_local.infra.gpu_optimization"


def test_gpu_optimization_root_shim_still_works() -> None:
    import gpu_optimization  # noqa: F401

    assert hasattr(gpu_optimization, "OptimizationLevel")
    assert hasattr(gpu_optimization, "FusionStrategy")
    assert hasattr(gpu_optimization, "GpuCapability")
    assert hasattr(gpu_optimization, "FusionConfig")
    assert hasattr(gpu_optimization, "BatchPreprocessor")
    assert hasattr(gpu_optimization, "GpuOptimizer")


def test_gpu_optimization_root_shim_private_symbols() -> None:
    import gpu_optimization  # noqa: F401

    # Module-level private constants and helpers must be exposed via the shim
    assert hasattr(gpu_optimization, "_BYTES_PER_FLOAT32")
    assert hasattr(gpu_optimization, "_MEMORY_OVERHEAD")
    assert hasattr(gpu_optimization, "_try_import_torch")
    assert hasattr(gpu_optimization, "_try_import_numpy")
    assert hasattr(gpu_optimization, "logger")


def test_gpu_optimization_from_import() -> None:
    from gpu_optimization import (
        BatchPreprocessor,
        FusionConfig,
        GpuOptimizer,
        OptimizationLevel,
    )

    assert isinstance(FusionConfig, type)
    assert isinstance(BatchPreprocessor, type)
    assert isinstance(GpuOptimizer, type)
    assert OptimizationLevel is not None


def test_gpu_optimization_root_and_native_are_same_object() -> None:
    import gpu_optimization

    native = importlib.import_module("ocr_local.infra.gpu_optimization")
    assert gpu_optimization is native


def test_gpu_optimization_not_in_infra_modules() -> None:
    from ocr_local.infra import _INFRA_MODULES

    assert "gpu_optimization" not in _INFRA_MODULES


def test_gpu_optimization_not_in_compat_finder() -> None:
    from ocr_local._compat_finder import _SUBPACKAGE_MODULES

    assert "gpu_optimization" not in _SUBPACKAGE_MODULES.get(
        "ocr_local.infra", frozenset()
    )


def test_gpu_optimization_infra_attr_access() -> None:
    import ocr_local.infra.gpu_optimization  # noqa: F401
    from ocr_local import infra

    mod = infra.gpu_optimization
    assert mod.__name__ == "ocr_local.infra.gpu_optimization"


# ---------------------------------------------------------------------------
# ocr_inference_backend -- Phase 2 migration tests (infra)
# ---------------------------------------------------------------------------


def test_ocr_inference_backend_native_subpackage_import() -> None:
    mod = importlib.import_module("ocr_local.infra.ocr_inference_backend")
    assert mod.__file__ is not None
    assert mod.__name__ == "ocr_local.infra.ocr_inference_backend"


def test_ocr_inference_backend_root_shim_still_works() -> None:
    import ocr_inference_backend  # noqa: F401

    assert hasattr(ocr_inference_backend, "INFERENCE_BACKEND")
    assert hasattr(ocr_inference_backend, "VALID_BACKENDS")
    assert hasattr(ocr_inference_backend, "get_backend")
    assert hasattr(ocr_inference_backend, "create_ocr_engine")
    assert hasattr(ocr_inference_backend, "get_backend_info")


def test_ocr_inference_backend_root_shim_private_symbols() -> None:
    import ocr_inference_backend  # noqa: F401

    # Module-level private helpers must be exposed via the shim
    assert hasattr(ocr_inference_backend, "_detect_best_backend")
    assert hasattr(ocr_inference_backend, "logger")


def test_ocr_inference_backend_from_import() -> None:
    from ocr_inference_backend import (
        VALID_BACKENDS,
        create_ocr_engine,
        get_backend,
    )

    assert callable(get_backend)
    assert callable(create_ocr_engine)
    assert isinstance(VALID_BACKENDS, tuple)


def test_ocr_inference_backend_root_and_native_are_same_object() -> None:
    import ocr_inference_backend

    native = importlib.import_module("ocr_local.infra.ocr_inference_backend")
    assert ocr_inference_backend is native


def test_ocr_inference_backend_not_in_infra_modules() -> None:
    from ocr_local.infra import _INFRA_MODULES

    assert "ocr_inference_backend" not in _INFRA_MODULES


def test_ocr_inference_backend_not_in_compat_finder() -> None:
    from ocr_local._compat_finder import _SUBPACKAGE_MODULES

    assert "ocr_inference_backend" not in _SUBPACKAGE_MODULES.get(
        "ocr_local.infra", frozenset()
    )


def test_ocr_inference_backend_infra_attr_access() -> None:
    import ocr_local.infra.ocr_inference_backend  # noqa: F401
    from ocr_local import infra

    mod = infra.ocr_inference_backend
    assert mod.__name__ == "ocr_local.infra.ocr_inference_backend"


# ---------------------------------------------------------------------------
# ocr_metrics -- Phase 2 migration tests (infra)
# ---------------------------------------------------------------------------


def test_ocr_metrics_native_subpackage_import() -> None:
    mod = importlib.import_module("ocr_local.infra.ocr_metrics")
    assert mod.__file__ is not None
    assert mod.__name__ == "ocr_local.infra.ocr_metrics"


def test_ocr_metrics_root_shim_still_works() -> None:
    import ocr_metrics  # noqa: F401

    assert hasattr(ocr_metrics, "ENGINE_USAGE_COUNTER")
    assert hasattr(ocr_metrics, "DPI_ESCALATION_COUNTER")
    assert hasattr(ocr_metrics, "record_engine_usage")
    assert hasattr(ocr_metrics, "record_dpi_escalation")


def test_ocr_metrics_root_shim_private_symbols() -> None:
    import ocr_metrics  # noqa: F401

    # Module-level private constants and helpers must be exposed via the shim
    assert hasattr(ocr_metrics, "_HAS_PROM")
    assert hasattr(ocr_metrics, "_VALID_ENGINES")
    assert hasattr(ocr_metrics, "_normalise_engine")
    assert hasattr(ocr_metrics, "logger")


def test_ocr_metrics_from_import() -> None:
    from ocr_metrics import (
        record_dpi_escalation,
        record_engine_usage,
    )

    assert callable(record_engine_usage)
    assert callable(record_dpi_escalation)


def test_ocr_metrics_root_and_native_are_same_object() -> None:
    import ocr_metrics

    native = importlib.import_module("ocr_local.infra.ocr_metrics")
    assert ocr_metrics is native


def test_ocr_metrics_not_in_infra_modules() -> None:
    from ocr_local.infra import _INFRA_MODULES

    assert "ocr_metrics" not in _INFRA_MODULES


def test_ocr_metrics_not_in_compat_finder() -> None:
    from ocr_local._compat_finder import _SUBPACKAGE_MODULES

    assert "ocr_metrics" not in _SUBPACKAGE_MODULES.get(
        "ocr_local.infra", frozenset()
    )


def test_ocr_metrics_infra_attr_access() -> None:
    import ocr_local.infra.ocr_metrics  # noqa: F401
    from ocr_local import infra

    mod = infra.ocr_metrics
    assert mod.__name__ == "ocr_local.infra.ocr_metrics"


# ---------------------------------------------------------------------------
# page_cache -- Phase 2 migration tests (infra)
# ---------------------------------------------------------------------------


def test_page_cache_native_subpackage_import() -> None:
    mod = importlib.import_module("ocr_local.infra.page_cache")
    assert mod.__file__ is not None
    assert mod.__name__ == "ocr_local.infra.page_cache"


def test_page_cache_root_shim_still_works() -> None:
    import page_cache  # noqa: F401

    assert hasattr(page_cache, "CacheStrategy")
    assert hasattr(page_cache, "CacheEntry")
    assert hasattr(page_cache, "CacheStats")
    assert hasattr(page_cache, "PageCache")


def test_page_cache_root_shim_private_symbols() -> None:
    import page_cache  # noqa: F401

    # Module-level non-``__all__`` attributes must be exposed via the shim
    assert hasattr(page_cache, "logger")


def test_page_cache_from_import() -> None:
    from page_cache import CacheEntry, CacheStats, CacheStrategy, PageCache

    assert isinstance(CacheEntry, type)
    assert isinstance(CacheStats, type)
    assert isinstance(PageCache, type)
    assert CacheStrategy is not None


def test_page_cache_root_and_native_are_same_object() -> None:
    import page_cache

    native = importlib.import_module("ocr_local.infra.page_cache")
    assert page_cache is native


def test_page_cache_not_in_infra_modules() -> None:
    from ocr_local.infra import _INFRA_MODULES

    assert "page_cache" not in _INFRA_MODULES


def test_page_cache_not_in_compat_finder() -> None:
    from ocr_local._compat_finder import _SUBPACKAGE_MODULES

    assert "page_cache" not in _SUBPACKAGE_MODULES.get(
        "ocr_local.infra", frozenset()
    )


def test_page_cache_infra_attr_access() -> None:
    import ocr_local.infra.page_cache  # noqa: F401
    from ocr_local import infra

    mod = infra.page_cache
    assert mod.__name__ == "ocr_local.infra.page_cache"


# ---------------------------------------------------------------------------
# page_routing -- Phase 2 migration tests (final infra migration)
# ---------------------------------------------------------------------------


def test_page_routing_native_subpackage_import() -> None:
    mod = importlib.import_module("ocr_local.infra.page_routing")
    assert mod.__file__ is not None
    assert mod.__name__ == "ocr_local.infra.page_routing"


def test_page_routing_root_shim_still_works() -> None:
    import page_routing  # noqa: F401

    assert hasattr(page_routing, "RoutingTarget")
    assert hasattr(page_routing, "PageFeatures")
    assert hasattr(page_routing, "RoutingDecision")
    assert hasattr(page_routing, "RoutingRule")
    assert hasattr(page_routing, "PageRouter")


def test_page_routing_root_shim_private_symbols() -> None:
    import page_routing  # noqa: F401

    # Module-level private helpers must be exposed via the shim
    assert hasattr(page_routing, "_estimate_duration")
    assert hasattr(page_routing, "_build_default_rules")
    assert hasattr(page_routing, "logger")


def test_page_routing_from_import() -> None:
    from page_routing import (
        PageFeatures,
        PageRouter,
        RoutingDecision,
        RoutingTarget,
    )

    assert isinstance(PageFeatures, type)
    assert isinstance(RoutingDecision, type)
    assert isinstance(PageRouter, type)
    assert RoutingTarget is not None


def test_page_routing_root_and_native_are_same_object() -> None:
    import page_routing

    native = importlib.import_module("ocr_local.infra.page_routing")
    assert page_routing is native


def test_page_routing_not_in_infra_modules() -> None:
    from ocr_local.infra import _INFRA_MODULES

    assert "page_routing" not in _INFRA_MODULES


def test_page_routing_not_in_compat_finder() -> None:
    from ocr_local._compat_finder import _SUBPACKAGE_MODULES

    assert "page_routing" not in _SUBPACKAGE_MODULES.get(
        "ocr_local.infra", frozenset()
    )


def test_page_routing_infra_attr_access() -> None:
    import ocr_local.infra.page_routing  # noqa: F401
    from ocr_local import infra

    mod = infra.page_routing
    assert mod.__name__ == "ocr_local.infra.page_routing"


# ---------------------------------------------------------------------------
# env_utils -- Phase 2 migration tests (config)
# ---------------------------------------------------------------------------


def test_env_utils_native_subpackage_import() -> None:
    mod = importlib.import_module("ocr_local.config.env_utils")
    assert mod.__file__ is not None
    assert mod.__name__ == "ocr_local.config.env_utils"


def test_env_utils_root_shim_still_works() -> None:
    import env_utils  # noqa: F401

    assert hasattr(env_utils, "get_env")
    assert hasattr(env_utils, "get_env_int")
    assert hasattr(env_utils, "get_env_float")
    assert hasattr(env_utils, "get_env_bool")
    assert hasattr(env_utils, "validate_env")
    assert hasattr(env_utils, "ENV_SCHEMA")


def test_env_utils_root_shim_private_symbols() -> None:
    import env_utils  # noqa: F401

    # Module-level public constants and callables must be exposed via the shim
    assert callable(env_utils.get_env)
    assert isinstance(env_utils.ENV_SCHEMA, dict)


def test_env_utils_from_import() -> None:
    from env_utils import (
        ENV_SCHEMA,
        get_env,
        get_env_bool,
        get_env_float,
        get_env_int,
        validate_env,
    )

    assert callable(get_env)
    assert callable(get_env_int)
    assert callable(get_env_float)
    assert callable(get_env_bool)
    assert callable(validate_env)
    assert isinstance(ENV_SCHEMA, dict)


def test_env_utils_root_and_native_are_same_object() -> None:
    import env_utils

    native = importlib.import_module("ocr_local.config.env_utils")
    assert env_utils is native


def test_env_utils_not_in_config_modules() -> None:
    from ocr_local.config import _CONFIG_MODULES

    assert "env_utils" not in _CONFIG_MODULES


def test_env_utils_not_in_compat_finder() -> None:
    from ocr_local._compat_finder import _SUBPACKAGE_MODULES

    assert "env_utils" not in _SUBPACKAGE_MODULES.get(
        "ocr_local.config", frozenset()
    )


def test_env_utils_config_attr_access() -> None:
    import ocr_local.config.env_utils  # noqa: F401
    from ocr_local import config

    mod = config.env_utils
    assert mod.__name__ == "ocr_local.config.env_utils"


# ---------------------------------------------------------------------------
# feature_flags -- Phase 2 migration tests (config)
# ---------------------------------------------------------------------------


def test_feature_flags_native_subpackage_import() -> None:
    mod = importlib.import_module("ocr_local.config.feature_flags")
    assert mod.__file__ is not None
    assert mod.__name__ == "ocr_local.config.feature_flags"


def test_feature_flags_root_shim_still_works() -> None:
    import feature_flags  # noqa: F401

    assert hasattr(feature_flags, "get_pipeline_features")
    assert hasattr(feature_flags, "is_feature_available")


def test_feature_flags_root_shim_private_symbols() -> None:
    import feature_flags  # noqa: F401

    # Module-level private constants and helpers must be exposed via the shim
    assert hasattr(feature_flags, "_FEATURE_MAP")
    assert hasattr(feature_flags, "_cache")
    assert hasattr(feature_flags, "logger")


def test_feature_flags_from_import() -> None:
    from feature_flags import get_pipeline_features, is_feature_available

    assert callable(get_pipeline_features)
    assert callable(is_feature_available)


def test_feature_flags_root_and_native_are_same_object() -> None:
    import feature_flags

    native = importlib.import_module("ocr_local.config.feature_flags")
    assert feature_flags is native


def test_feature_flags_not_in_config_modules() -> None:
    from ocr_local.config import _CONFIG_MODULES

    assert "feature_flags" not in _CONFIG_MODULES


def test_feature_flags_not_in_compat_finder() -> None:
    from ocr_local._compat_finder import _SUBPACKAGE_MODULES

    assert "feature_flags" not in _SUBPACKAGE_MODULES.get(
        "ocr_local.config", frozenset()
    )


def test_feature_flags_config_attr_access() -> None:
    import ocr_local.config.feature_flags  # noqa: F401
    from ocr_local import config

    mod = config.feature_flags
    assert mod.__name__ == "ocr_local.config.feature_flags"


# ---------------------------------------------------------------------------
# language_config -- Phase 2 migration tests (config)
# ---------------------------------------------------------------------------


def test_language_config_native_subpackage_import() -> None:
    mod = importlib.import_module("ocr_local.config.language_config")
    assert mod.__file__ is not None
    assert mod.__name__ == "ocr_local.config.language_config"


def test_language_config_root_shim_still_works() -> None:
    import language_config  # noqa: F401

    assert hasattr(language_config, "LanguageEntry")
    assert hasattr(language_config, "LANGUAGE_REGISTRY")
    assert hasattr(language_config, "LANG_MAPPING")
    assert hasattr(language_config, "TARGET_LANGS")
    assert hasattr(language_config, "FONT_MAP")
    assert hasattr(language_config, "TESSERACT_MAP")
    assert hasattr(language_config, "EASYOCR_MAP")
    assert hasattr(language_config, "RTL_LANGUAGES")


def test_language_config_root_shim_private_symbols() -> None:
    import language_config  # noqa: F401

    # Module-level private helpers must be exposed via the shim
    assert hasattr(language_config, "_reg")
    assert hasattr(language_config, "logger")


def test_language_config_from_import() -> None:
    from language_config import (
        LANG_MAPPING,
        LANGUAGE_REGISTRY,
        RTL_LANGUAGES,
        TARGET_LANGS,
        LanguageEntry,
        get_paddle_code,
        get_tesseract_code,
        is_rtl,
    )

    assert isinstance(LanguageEntry, type)
    assert isinstance(LANGUAGE_REGISTRY, dict)
    assert isinstance(LANG_MAPPING, dict)
    assert isinstance(TARGET_LANGS, list)
    assert isinstance(RTL_LANGUAGES, set)
    assert callable(get_paddle_code)
    assert callable(get_tesseract_code)
    assert callable(is_rtl)


def test_language_config_root_and_native_are_same_object() -> None:
    import language_config

    native = importlib.import_module("ocr_local.config.language_config")
    assert language_config is native


def test_language_config_not_in_config_modules() -> None:
    from ocr_local.config import _CONFIG_MODULES

    assert "language_config" not in _CONFIG_MODULES


def test_language_config_not_in_compat_finder() -> None:
    from ocr_local._compat_finder import _SUBPACKAGE_MODULES

    assert "language_config" not in _SUBPACKAGE_MODULES.get(
        "ocr_local.config", frozenset()
    )


def test_language_config_config_attr_access() -> None:
    import ocr_local.config.language_config  # noqa: F401
    from ocr_local import config

    mod = config.language_config
    assert mod.__name__ == "ocr_local.config.language_config"


# ---------------------------------------------------------------------------
# version -- Phase 2 migration tests (final config migration)
# ---------------------------------------------------------------------------


def test_version_native_subpackage_import() -> None:
    mod = importlib.import_module("ocr_local.config.version")
    assert mod.__file__ is not None
    assert mod.__name__ == "ocr_local.config.version"


def test_version_root_shim_still_works() -> None:
    import version  # noqa: F401

    assert hasattr(version, "__version__")
    assert hasattr(version, "__version_info__")


def test_version_root_shim_public_symbols() -> None:
    import version  # noqa: F401

    # __version__ must be a dotted version string
    assert isinstance(version.__version__, str)
    assert "." in version.__version__
    # __version_info__ must be a tuple of ints
    assert isinstance(version.__version_info__, tuple)
    assert all(isinstance(x, int) for x in version.__version_info__)


def test_version_from_import() -> None:
    from version import __version__, __version_info__

    assert isinstance(__version__, str)
    assert isinstance(__version_info__, tuple)


def test_version_root_and_native_are_same_object() -> None:
    import version

    native = importlib.import_module("ocr_local.config.version")
    assert version is native


def test_version_not_in_config_modules() -> None:
    from ocr_local.config import _CONFIG_MODULES

    assert "version" not in _CONFIG_MODULES


def test_version_not_in_compat_finder() -> None:
    from ocr_local._compat_finder import _SUBPACKAGE_MODULES

    assert "version" not in _SUBPACKAGE_MODULES.get(
        "ocr_local.config", frozenset()
    )


def test_version_config_attr_access() -> None:
    import ocr_local.config.version  # noqa: F401
    from ocr_local import config

    mod = config.version
    assert mod.__name__ == "ocr_local.config.version"
