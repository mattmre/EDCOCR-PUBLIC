"""
pipeline_config.py -- Typed configuration dataclass for the OCR pipeline.

Centralises the pipeline parameters that were previously scattered as
module-level globals in ``ocr_gpu_async.py``.  The module globals remain in
place for backward compatibility; this dataclass is the path toward removing
them in a future session.

Phase 1 added 21 numeric/simple parameters (Groups 1-6).
Phase 2 added 30 feature-flag fields (Groups 7-12) covering Document
Intelligence, form/KV extraction, custody, processing features, analysis
features, and pipeline optimization toggles.

Usage:
    from pipeline_config import PipelineConfig, create_pipeline_config

    cfg = create_pipeline_config()       # reads from os.environ
    cfg = create_pipeline_config({})     # explicit empty dict = all defaults
    cfg = PipelineConfig(dpi=450)        # override specific fields directly

Design notes:
    - NOT frozen: ``main()`` in ``ocr_gpu_async.py`` overrides fields at
      startup via CLI arguments.  Mutability is therefore required.
    - Validation is performed in ``__post_init__`` with descriptive
      ``ValueError`` messages on out-of-range values.
    - The ``create_pipeline_config`` factory uses local ``_int``/``_float``/
      ``_bool`` helpers rather than importing from ``ocr_gpu_async`` to
      avoid circular imports -- ``ocr_gpu_async`` imports *this* module.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Mapping, Optional

__all__ = ["PipelineConfig", "create_pipeline_config"]


_TRUTHY = ("1", "true", "yes")

_DOCINTEL_MODES = frozenset({"layout_only", "tables_only", "full"})
_PREPROCESSING_LEVELS = frozenset({"none", "standard", "enhanced", "aggressive"})
_EXTRACTOR_MODES = frozenset({"thread", "process", "auto"})
_LANGUAGE_REDACT_MODES = frozenset({"true", "false", "privilege_or_short_doc"})
_TRANSLATION_QUALITY_LEVELS = frozenset({"draft", "standard", "legal"})
_TRANSLATION_LATENCY_LEVELS = frozenset({"realtime", "standard", "bulk"})
_ALLOWED_PATH_PREFIXES = ("/app/",)


@dataclasses.dataclass
class PipelineConfig:
    """Typed configuration for the async OCR pipeline.

    Fields are grouped into 12 logical sections (6 numeric/path groups from
    Phase 1 plus 6 feature-flag groups added in Phase 2) -- see ``__repr__``
    for the human-readable breakdown used in startup logging.
    """

    # --- Group 1: Queue sizes ---------------------------------------------
    image_queue_size: int = 200
    chunk_queue_size: int = 50
    result_queue_size: int = 5000
    compression_queue_size: int = 5000

    # --- Group 2: Thread counts -------------------------------------------
    num_extractors: int = 8
    num_workers: int = 12
    num_compressors: int = 8
    num_assembler_workers: int = 4
    pdf_conversion_threads: int = 1
    extractor_process_workers: int = 8
    extractor_mode: str = "thread"

    # --- Group 3: Timeouts ------------------------------------------------
    poppler_timeout: int = 300
    tesseract_timeout: int = 120
    thread_join_timeout: int = 30
    shutdown_drain_timeout_seconds: int = 300

    # --- Group 4: Processing params ---------------------------------------
    dpi: int = 300
    jpeg_quality: int = 85
    monitor_sleep_seconds: int = 10
    keep_temp_files: bool = False
    chunk_target_size: int = 20

    # --- Group 5: Video params --------------------------------------------
    video_frame_sample_seconds: float = 1.0
    video_max_frames: int = 300

    # --- Group 6: ML + runtime paths --------------------------------------
    fasttext_model_path: str = "/app/models/lid.176.bin"
    source_folder: str = "/app/ocr_source"
    output_folder: str = "/app/ocr_output"
    temp_folder: str = "/app/ocr_temp"
    log_dir: str = "/app/ocr_output/logs"
    failure_report: str = "/app/ocr_output/failures.csv"
    healthcheck_file: str = "/app/ocr_healthcheck"

    # --- Group 7: Document Intelligence (Phase 3A) ------------------------
    enable_document_intelligence: bool = False
    enable_layout_analysis: bool = True
    enable_table_extraction: bool = True
    docintel_mode: str = "full"
    export_tables: bool = False

    # --- Group 8: Form & KV Extraction (Phase 3C) -------------------------
    enable_form_detection: bool = False
    enable_kv_extraction: bool = False
    enable_privilege_detection: bool = False

    # --- Group 9: Custody & Forensics -------------------------------------
    enable_custody: bool = True

    # --- Group 10: Processing Features ------------------------------------
    enable_preprocessing: bool = False
    preprocessing_level: str = "standard"
    enable_noise_profiling: bool = False
    enable_validation: bool = True
    enable_dpi_escalation: bool = False
    dpi_confidence_threshold: float = 0.60

    # --- Group 11: Analysis Features --------------------------------------
    enable_ner: bool = False
    enable_handwriting: bool = False
    enable_signature_verification: bool = False
    enable_vertical_text: bool = False
    enable_table_fallback: bool = False
    enable_classification: bool = False
    enable_extraction: bool = False
    enable_specialist_routing: bool = False
    enable_entity_consolidation: bool = False
    enable_relationship_extraction: bool = False
    enable_retrieval_output: bool = False
    enable_exception_routing: bool = False

    # --- Group 12: Pipeline Optimization ----------------------------------
    enable_adaptive_batch: bool = False
    enable_page_cache: bool = False
    enable_page_routing: bool = False
    enable_gpu_optimization: bool = False

    # --- Group 13: Per-span language detection (Plan A) -------------------
    enable_per_span_language: bool = False
    language_include_spans: bool = False
    language_short_span_threshold: int = 20
    language_confidence_threshold: float = 0.4
    language_redact_samples: str = "privilege_or_short_doc"

    # --- Group 14: Translation enrichment (Plan B Wave M1) ----------------
    # NEVER flip ``enable_translation`` to True without an explicit 48h
    # bake -- the assembler hook is fail-open but enabling it changes
    # which custody events the audit trail captures.
    enable_translation: bool = False
    translation_target_languages: list = dataclasses.field(default_factory=list)
    translation_quality: str = "standard"
    translation_latency: str = "standard"
    enable_handwriting_mt: bool = False  # Plan B Q11
    # Phase 3 EDC split seam.  Defaults keep the existing in-repo
    # translation path unchanged; the external EDC_TRANSLATION service is
    # used only when explicitly preferred by config or env.
    translation_prefer_external_service: bool = False
    translation_external_service_url: Optional[str] = None
    translation_external_provider_id: str = "passthrough"
    translation_external_timeout_seconds: float = 30.0
    translation_external_readiness_path: str = "/health"

    # --- Group 15: Translation model cache (Plan B Wave M2) ---------------
    # ``translation_cache_dir`` is None -> use the default
    # ``~/.cache/ocr_local/translation/``.  ``translation_airgapped``
    # forces the cache resolver to refuse downloads even when the
    # caller passes ``allow_download=True`` -- pre-baked models only.
    translation_cache_dir: Optional[str] = None
    translation_cache_max_bytes: int = 50 * 1024 * 1024 * 1024
    translation_airgapped: bool = False

    # --- Group 16: Batch translation scheduling (Plan B Wave M2 -- B17) ---
    # All defaults disabled / safe.  ``translation_batch_enabled`` gates
    # both the submit_batch helper and the API router registration.  The
    # other knobs cap memory/concurrency to prevent a single tenant from
    # saturating the translation_batch queue.
    translation_batch_enabled: bool = False
    translation_batch_max_inputs: int = 1000
    translation_batch_input_max_bytes: int = 8 * 1024
    translation_batch_fan_out_size: int = 32
    translation_batch_concurrency: int = 4

    # --- Group 17: Translation model provenance (Plan B Wave M2 -- B19) ---
    # ``translation_enforce_provenance`` gates the provenance validator
    # in :mod:`ocr_local.translation.provenance`.  When True the engine
    # registry refuses to bind any MT model whose ``model_provenance()``
    # is missing the SLSA v1.0 / in-toto / SBOM fields required by
    # E-B-008 (RED-07).  Default False so the existing OPUS-MT stub and
    # legacy test fixtures keep working; production deployments must
    # flip this on alongside ``enable_translation``.
    translation_enforce_provenance: bool = False

    def __post_init__(self) -> None:
        """Validate all numeric ranges after construction."""
        # Group 1: queue sizes
        _require_range("image_queue_size", self.image_queue_size, 1, 10_000)
        _require_range("chunk_queue_size", self.chunk_queue_size, 1, 10_000)
        _require_range("result_queue_size", self.result_queue_size, 1, 100_000)
        _require_range(
            "compression_queue_size", self.compression_queue_size, 1, 100_000
        )

        # Group 2: thread counts (all must be >= 1)
        _require_range("num_extractors", self.num_extractors, 1, 64)
        _require_range("num_workers", self.num_workers, 1, 64)
        _require_range("num_compressors", self.num_compressors, 1, 64)
        _require_range("num_assembler_workers", self.num_assembler_workers, 1, 32)
        _require_range("pdf_conversion_threads", self.pdf_conversion_threads, 1, 16)
        _require_range(
            "extractor_process_workers", self.extractor_process_workers, 1, 64
        )
        if self.extractor_mode not in _EXTRACTOR_MODES:
            raise ValueError(
                "extractor_mode must be one of "
                f"{sorted(_EXTRACTOR_MODES)}, got {self.extractor_mode!r}"
            )

        # Group 3: timeouts
        _require_range("poppler_timeout", self.poppler_timeout, 1, 1800)
        _require_range("tesseract_timeout", self.tesseract_timeout, 1, 600)
        _require_range("thread_join_timeout", self.thread_join_timeout, 1, 600)
        _require_range(
            "shutdown_drain_timeout_seconds",
            self.shutdown_drain_timeout_seconds,
            1,
            3600,
        )

        # Group 4: processing params
        if not (72 <= self.dpi <= 1200):
            raise ValueError(f"DPI must be 72..1200, got {self.dpi}")
        if not (1 <= self.jpeg_quality <= 100):
            raise ValueError(
                f"jpeg_quality must be 1..100, got {self.jpeg_quality}"
            )
        _require_range(
            "monitor_sleep_seconds", self.monitor_sleep_seconds, 1, 3600
        )
        if not isinstance(self.keep_temp_files, bool):
            raise ValueError(
                f"keep_temp_files must be bool, got {type(self.keep_temp_files).__name__}"
            )
        _require_range("chunk_target_size", self.chunk_target_size, 1, 500)

        # Group 5: video params
        if not (0.1 <= float(self.video_frame_sample_seconds) <= 600.0):
            raise ValueError(
                "video_frame_sample_seconds must be 0.1..600.0, got "
                f"{self.video_frame_sample_seconds}"
            )
        _require_range("video_max_frames", self.video_max_frames, 1, 10_000)

        # Group 6: ML paths
        if not isinstance(self.fasttext_model_path, str) or not self.fasttext_model_path:
            raise ValueError(
                "fasttext_model_path must be a non-empty string, got "
                f"{self.fasttext_model_path!r}"
            )
        for name in (
            "source_folder",
            "output_folder",
            "temp_folder",
            "log_dir",
            "failure_report",
            "healthcheck_file",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"{name} must be a non-empty string, got {value!r}"
                )

        # Group 7: Document Intelligence
        _require_bool("enable_document_intelligence", self.enable_document_intelligence)
        _require_bool("enable_layout_analysis", self.enable_layout_analysis)
        _require_bool("enable_table_extraction", self.enable_table_extraction)
        if self.docintel_mode not in _DOCINTEL_MODES:
            raise ValueError(
                "docintel_mode must be one of "
                f"{sorted(_DOCINTEL_MODES)}, got {self.docintel_mode!r}"
            )
        _require_bool("export_tables", self.export_tables)

        # Group 8: Form & KV extraction
        _require_bool("enable_form_detection", self.enable_form_detection)
        _require_bool("enable_kv_extraction", self.enable_kv_extraction)
        _require_bool("enable_privilege_detection", self.enable_privilege_detection)

        # Group 9: Custody
        _require_bool("enable_custody", self.enable_custody)

        # Group 10: Processing features
        _require_bool("enable_preprocessing", self.enable_preprocessing)
        if self.preprocessing_level not in _PREPROCESSING_LEVELS:
            raise ValueError(
                "preprocessing_level must be one of "
                f"{sorted(_PREPROCESSING_LEVELS)}, got {self.preprocessing_level!r}"
            )
        _require_bool("enable_noise_profiling", self.enable_noise_profiling)
        _require_bool("enable_validation", self.enable_validation)
        _require_bool("enable_dpi_escalation", self.enable_dpi_escalation)
        if not isinstance(self.dpi_confidence_threshold, (int, float)) or isinstance(
            self.dpi_confidence_threshold, bool
        ):
            raise ValueError(
                "dpi_confidence_threshold must be float, got "
                f"{type(self.dpi_confidence_threshold).__name__}"
            )
        if not (0.0 <= float(self.dpi_confidence_threshold) <= 1.0):
            raise ValueError(
                "dpi_confidence_threshold must be 0.0..1.0, got "
                f"{self.dpi_confidence_threshold}"
            )

        # Group 11: Analysis features
        _require_bool("enable_ner", self.enable_ner)
        _require_bool("enable_handwriting", self.enable_handwriting)
        _require_bool(
            "enable_signature_verification", self.enable_signature_verification
        )
        _require_bool("enable_vertical_text", self.enable_vertical_text)
        _require_bool("enable_table_fallback", self.enable_table_fallback)
        _require_bool("enable_classification", self.enable_classification)
        _require_bool("enable_extraction", self.enable_extraction)
        _require_bool("enable_specialist_routing", self.enable_specialist_routing)
        _require_bool(
            "enable_entity_consolidation", self.enable_entity_consolidation
        )
        _require_bool(
            "enable_relationship_extraction", self.enable_relationship_extraction
        )
        _require_bool("enable_retrieval_output", self.enable_retrieval_output)
        _require_bool("enable_exception_routing", self.enable_exception_routing)

        # Group 12: Pipeline optimization
        _require_bool("enable_adaptive_batch", self.enable_adaptive_batch)
        _require_bool("enable_page_cache", self.enable_page_cache)
        _require_bool("enable_page_routing", self.enable_page_routing)
        _require_bool("enable_gpu_optimization", self.enable_gpu_optimization)

        # Group 13: Per-span language detection
        _require_bool("enable_per_span_language", self.enable_per_span_language)
        _require_bool("language_include_spans", self.language_include_spans)
        _require_range(
            "language_short_span_threshold",
            self.language_short_span_threshold,
            1,
            10_000,
        )
        if not isinstance(self.language_confidence_threshold, (int, float)) or isinstance(
            self.language_confidence_threshold, bool
        ):
            raise ValueError(
                "language_confidence_threshold must be float, got "
                f"{type(self.language_confidence_threshold).__name__}"
            )
        if not (0.0 <= float(self.language_confidence_threshold) <= 1.0):
            raise ValueError(
                "language_confidence_threshold must be 0.0..1.0, got "
                f"{self.language_confidence_threshold}"
            )
        if self.language_redact_samples not in _LANGUAGE_REDACT_MODES:
            raise ValueError(
                "language_redact_samples must be one of "
                f"{sorted(_LANGUAGE_REDACT_MODES)}, got "
                f"{self.language_redact_samples!r}"
            )

        # Group 14: Translation
        _require_bool("enable_translation", self.enable_translation)
        if not isinstance(self.translation_target_languages, list):
            raise ValueError(
                "translation_target_languages must be list, got "
                f"{type(self.translation_target_languages).__name__}"
            )
        for tgt in self.translation_target_languages:
            if not isinstance(tgt, str) or not tgt:
                raise ValueError(
                    "translation_target_languages entries must be non-empty "
                    f"strings, got {tgt!r}"
                )
        if self.translation_quality not in _TRANSLATION_QUALITY_LEVELS:
            raise ValueError(
                "translation_quality must be one of "
                f"{sorted(_TRANSLATION_QUALITY_LEVELS)}, got "
                f"{self.translation_quality!r}"
            )
        if self.translation_latency not in _TRANSLATION_LATENCY_LEVELS:
            raise ValueError(
                "translation_latency must be one of "
                f"{sorted(_TRANSLATION_LATENCY_LEVELS)}, got "
                f"{self.translation_latency!r}"
            )
        _require_bool("enable_handwriting_mt", self.enable_handwriting_mt)
        _require_bool(
            "translation_prefer_external_service",
            self.translation_prefer_external_service,
        )
        if self.translation_external_service_url is not None and not isinstance(
            self.translation_external_service_url,
            str,
        ):
            raise ValueError(
                "translation_external_service_url must be str or None, got "
                f"{type(self.translation_external_service_url).__name__}"
            )
        if (
            self.translation_external_service_url is not None
            and not self.translation_external_service_url.strip()
        ):
            raise ValueError(
                "translation_external_service_url must be a non-empty string"
            )
        if (
            not isinstance(self.translation_external_provider_id, str)
            or not self.translation_external_provider_id.strip()
        ):
            raise ValueError(
                "translation_external_provider_id must be a non-empty string"
            )
        if not isinstance(
            self.translation_external_timeout_seconds,
            (int, float),
        ) or isinstance(self.translation_external_timeout_seconds, bool):
            raise ValueError(
                "translation_external_timeout_seconds must be numeric, got "
                f"{type(self.translation_external_timeout_seconds).__name__}"
            )
        if not (0.1 <= float(self.translation_external_timeout_seconds) <= 600.0):
            raise ValueError(
                "translation_external_timeout_seconds must be 0.1..600.0, got "
                f"{self.translation_external_timeout_seconds}"
            )
        if (
            not isinstance(self.translation_external_readiness_path, str)
            or not self.translation_external_readiness_path.startswith("/")
        ):
            raise ValueError(
                "translation_external_readiness_path must be an absolute HTTP path"
            )

        # Group 15: Translation model cache (Plan B Wave M2)
        if self.translation_cache_dir is not None and not isinstance(
            self.translation_cache_dir, str
        ):
            raise ValueError(
                "translation_cache_dir must be str or None, got "
                f"{type(self.translation_cache_dir).__name__}"
            )
        if self.translation_cache_dir is not None and not self.translation_cache_dir:
            raise ValueError(
                "translation_cache_dir must be a non-empty string, got "
                f"{self.translation_cache_dir!r}"
            )
        _require_range(
            "translation_cache_max_bytes",
            self.translation_cache_max_bytes,
            1,
            10 * 1024 * 1024 * 1024 * 1024,  # 10 TB ceiling
        )
        _require_bool("translation_airgapped", self.translation_airgapped)

        # Group 16: Batch translation scheduling (Plan B Wave M2 -- B17)
        _require_bool(
            "translation_batch_enabled", self.translation_batch_enabled
        )
        _require_range(
            "translation_batch_max_inputs",
            self.translation_batch_max_inputs,
            1,
            100_000,
        )
        _require_range(
            "translation_batch_input_max_bytes",
            self.translation_batch_input_max_bytes,
            1,
            10 * 1024 * 1024,  # 10 MB ceiling per input
        )
        _require_range(
            "translation_batch_fan_out_size",
            self.translation_batch_fan_out_size,
            1,
            1024,
        )
        _require_range(
            "translation_batch_concurrency",
            self.translation_batch_concurrency,
            1,
            128,
        )

        # Group 17: Translation model provenance (Plan B Wave M2 -- B19)
        _require_bool(
            "translation_enforce_provenance",
            self.translation_enforce_provenance,
        )

    def __repr__(self) -> str:  # pragma: no cover - formatting only
        return (
            "PipelineConfig(\n"
            "  Queue sizes:\n"
            f"    image_queue_size       = {self.image_queue_size}\n"
            f"    chunk_queue_size       = {self.chunk_queue_size}\n"
            f"    result_queue_size      = {self.result_queue_size}\n"
            f"    compression_queue_size = {self.compression_queue_size}\n"
            "  Threads:\n"
            f"    num_extractors            = {self.num_extractors}\n"
            f"    num_workers               = {self.num_workers}\n"
            f"    num_compressors           = {self.num_compressors}\n"
            f"    num_assembler_workers     = {self.num_assembler_workers}\n"
            f"    pdf_conversion_threads    = {self.pdf_conversion_threads}\n"
            f"    extractor_process_workers = {self.extractor_process_workers}\n"
            f"    extractor_mode            = {self.extractor_mode}\n"
            "  Timeouts:\n"
            f"    poppler_timeout     = {self.poppler_timeout}s\n"
            f"    tesseract_timeout   = {self.tesseract_timeout}s\n"
            f"    thread_join_timeout = {self.thread_join_timeout}s\n"
            f"    shutdown_drain_timeout_seconds = "
            f"{self.shutdown_drain_timeout_seconds}s\n"
            "  Processing:\n"
            f"    dpi                   = {self.dpi}\n"
            f"    jpeg_quality          = {self.jpeg_quality}\n"
            f"    monitor_sleep_seconds = {self.monitor_sleep_seconds}\n"
            f"    keep_temp_files       = {self.keep_temp_files}\n"
            f"    chunk_target_size     = {self.chunk_target_size}\n"
            "  Video:\n"
            f"    video_frame_sample_seconds = {self.video_frame_sample_seconds}\n"
            f"    video_max_frames           = {self.video_max_frames}\n"
            "  ML + runtime paths:\n"
            f"    fasttext_model_path = {self.fasttext_model_path}\n"
            f"    source_folder       = {self.source_folder}\n"
            f"    output_folder       = {self.output_folder}\n"
            f"    temp_folder         = {self.temp_folder}\n"
            f"    log_dir             = {self.log_dir}\n"
            f"    failure_report      = {self.failure_report}\n"
            f"    healthcheck_file    = {self.healthcheck_file}\n"
            "  Document Intelligence:\n"
            f"    enable_document_intelligence = {self.enable_document_intelligence}\n"
            f"    enable_layout_analysis       = {self.enable_layout_analysis}\n"
            f"    enable_table_extraction      = {self.enable_table_extraction}\n"
            f"    docintel_mode                = {self.docintel_mode}\n"
            f"    export_tables                = {self.export_tables}\n"
            "  Form & KV:\n"
            f"    enable_form_detection      = {self.enable_form_detection}\n"
            f"    enable_kv_extraction       = {self.enable_kv_extraction}\n"
            f"    enable_privilege_detection = {self.enable_privilege_detection}\n"
            "  Custody:\n"
            f"    enable_custody = {self.enable_custody}\n"
            "  Processing features:\n"
            f"    enable_preprocessing     = {self.enable_preprocessing}\n"
            f"    preprocessing_level      = {self.preprocessing_level}\n"
            f"    enable_noise_profiling   = {self.enable_noise_profiling}\n"
            f"    enable_validation        = {self.enable_validation}\n"
            f"    enable_dpi_escalation    = {self.enable_dpi_escalation}\n"
            f"    dpi_confidence_threshold = {self.dpi_confidence_threshold}\n"
            "  Analysis features:\n"
            f"    enable_ner                     = {self.enable_ner}\n"
            f"    enable_handwriting             = {self.enable_handwriting}\n"
            f"    enable_signature_verification  = {self.enable_signature_verification}\n"
            f"    enable_vertical_text           = {self.enable_vertical_text}\n"
            f"    enable_table_fallback          = {self.enable_table_fallback}\n"
            f"    enable_classification          = {self.enable_classification}\n"
            f"    enable_extraction              = {self.enable_extraction}\n"
            f"    enable_specialist_routing      = {self.enable_specialist_routing}\n"
            f"    enable_entity_consolidation    = {self.enable_entity_consolidation}\n"
            f"    enable_relationship_extraction = {self.enable_relationship_extraction}\n"
            f"    enable_retrieval_output        = {self.enable_retrieval_output}\n"
            f"    enable_exception_routing       = {self.enable_exception_routing}\n"
            "  Pipeline optimization:\n"
            f"    enable_adaptive_batch   = {self.enable_adaptive_batch}\n"
            f"    enable_page_cache       = {self.enable_page_cache}\n"
            f"    enable_page_routing     = {self.enable_page_routing}\n"
            f"    enable_gpu_optimization = {self.enable_gpu_optimization}\n"
            ")"
        )


def _require_range(name: str, value: int, min_val: int, max_val: int) -> None:
    """Raise ``ValueError`` if ``value`` is outside ``[min_val, max_val]``."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(
            f"{name} must be int, got {type(value).__name__}"
        )
    if value < min_val or value > max_val:
        raise ValueError(
            f"{name} must be {min_val}..{max_val}, got {value}"
        )


def _require_bool(name: str, value: object) -> None:
    """Raise ``ValueError`` if ``value`` is not a ``bool``."""
    if not isinstance(value, bool):
        raise ValueError(
            f"{name} must be bool, got {type(value).__name__}"
        )


def _int(
    env: Mapping[str, str],
    key: str,
    default: int,
    *,
    min_val: int,
    max_val: int,
) -> int:
    """Parse an env var as int, clamping to ``[min_val, max_val]``.

    Invalid values silently fall back to ``default`` so that configuration
    errors never crash the pipeline at startup -- matching the behaviour of
    ``ocr_distributed.ocr_utils.get_env_int``.
    """
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    if value < min_val:
        return min_val
    if value > max_val:
        return max_val
    return value


def _float(
    env: Mapping[str, str],
    key: str,
    default: float,
    *,
    min_val: float,
    max_val: float,
) -> float:
    """Parse an env var as float, clamping to ``[min_val, max_val]``."""
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if value < min_val:
        return min_val
    if value > max_val:
        return max_val
    return value


def _bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    """Parse an env var as a truthy/falsy flag.

    Accepts ``"1"``, ``"true"``, ``"yes"`` (case-insensitive) as True.
    Empty or missing values fall back to ``default``.
    """
    raw = env.get(key)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized == "":
        return default
    return normalized in _TRUTHY


def _str_choice(
    env: Mapping[str, str],
    key: str,
    default: str,
    choices: frozenset,
) -> str:
    """Parse an env var as a constrained string choice.

    Returns ``default`` when the env var is missing, empty, or not in the
    allowed ``choices`` set.  Comparison is case-insensitive on the input
    side; values are stored lower-cased.
    """
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    val = raw.strip().lower()
    return val if val in choices else default


def _csv_list(env: Mapping[str, str], key: str) -> list[str]:
    """Parse a comma-separated env var into a clean list of strings.

    Empty/missing -> empty list.  Blank entries (e.g. trailing comma) are
    discarded.  Used for ``TRANSLATION_TARGET_LANGUAGES``.
    """
    raw = env.get(key)
    if raw is None or raw == "":
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _path(
    env: Mapping[str, str],
    key: str,
    default: str,
    allowed_prefixes: tuple[str, ...],
) -> str:
    """Parse a path env var, falling back to the default outside allowed roots."""
    raw = env.get(key, default)
    resolved = os.path.realpath(raw)
    if allowed_prefixes and not any(
        resolved.startswith(os.path.realpath(prefix))
        for prefix in allowed_prefixes
    ):
        return os.path.realpath(default)
    return resolved


def create_pipeline_config(
    env: Optional[Mapping[str, str]] = None,
) -> PipelineConfig:
    """Build a :class:`PipelineConfig` from environment variables.

    Parameters
    ----------
    env:
        Optional mapping of environment variables.  When ``None``, reads
        from :data:`os.environ`.  Passing an explicit dict (including the
        empty dict ``{}``) enables deterministic testing without mutating
        the process environment.
    """
    e: Mapping[str, str] = env if env is not None else os.environ
    source_folder = _path(
        e,
        "SOURCE_FOLDER",
        "/app/ocr_source",
        _ALLOWED_PATH_PREFIXES,
    )
    output_folder = _path(
        e,
        "OUTPUT_FOLDER",
        "/app/ocr_output",
        _ALLOWED_PATH_PREFIXES,
    )
    temp_folder = _path(
        e,
        "TEMP_FOLDER",
        "/app/ocr_temp",
        _ALLOWED_PATH_PREFIXES,
    )
    log_dir = _path(
        e,
        "LOG_DIR",
        os.path.join(output_folder, "logs"),
        _ALLOWED_PATH_PREFIXES,
    )
    failure_report = _path(
        e,
        "FAILURE_REPORT",
        os.path.join(output_folder, "failures.csv"),
        _ALLOWED_PATH_PREFIXES,
    )
    healthcheck_file = _path(
        e,
        "HEALTHCHECK_FILE",
        "/app/ocr_healthcheck",
        _ALLOWED_PATH_PREFIXES,
    )

    return PipelineConfig(
        # Group 1: queue sizes
        image_queue_size=_int(e, "IMAGE_QUEUE_SIZE", 200, min_val=1, max_val=10_000),
        chunk_queue_size=_int(e, "CHUNK_QUEUE_SIZE", 50, min_val=1, max_val=10_000),
        result_queue_size=_int(
            e, "RESULT_QUEUE_SIZE", 5000, min_val=1, max_val=100_000
        ),
        compression_queue_size=_int(
            e, "COMPRESSION_QUEUE_SIZE", 5000, min_val=1, max_val=100_000
        ),
        # Group 2: thread counts
        num_extractors=_int(e, "NUM_EXTRACTORS", 8, min_val=1, max_val=64),
        num_workers=_int(e, "NUM_WORKERS", 12, min_val=1, max_val=64),
        num_compressors=_int(e, "NUM_COMPRESSORS", 8, min_val=1, max_val=64),
        num_assembler_workers=_int(
            e, "NUM_ASSEMBLER_WORKERS", 4, min_val=1, max_val=32
        ),
        pdf_conversion_threads=_int(
            e, "PDF_CONVERSION_THREADS", 1, min_val=1, max_val=16
        ),
        extractor_process_workers=_int(
            e, "EXTRACTOR_PROCESS_WORKERS", 8, min_val=1, max_val=64
        ),
        extractor_mode=_str_choice(
            e, "EXTRACTOR_MODE", "thread", _EXTRACTOR_MODES
        ),
        # Group 3: timeouts
        poppler_timeout=_int(e, "POPPLER_TIMEOUT", 300, min_val=1, max_val=1800),
        tesseract_timeout=_int(
            e, "TESSERACT_TIMEOUT", 120, min_val=1, max_val=600
        ),
        thread_join_timeout=_int(
            e, "THREAD_JOIN_TIMEOUT", 30, min_val=1, max_val=600
        ),
        shutdown_drain_timeout_seconds=_int(
            e, "SHUTDOWN_DRAIN_TIMEOUT", 300, min_val=1, max_val=3600
        ),
        # Group 4: processing params
        dpi=_int(e, "DPI", 300, min_val=72, max_val=1200),
        jpeg_quality=_int(e, "JPEG_QUALITY", 85, min_val=1, max_val=100),
        monitor_sleep_seconds=_int(
            e, "MONITOR_SLEEP_SECONDS", 10, min_val=1, max_val=3600
        ),
        keep_temp_files=_bool(e, "KEEP_TEMP_FILES", False),
        chunk_target_size=_int(
            e, "CHUNK_TARGET_SIZE", 20, min_val=1, max_val=500
        ),
        # Group 5: video params
        video_frame_sample_seconds=_float(
            e,
            "VIDEO_FRAME_SAMPLE_SECONDS",
            1.0,
            min_val=0.1,
            max_val=600.0,
        ),
        video_max_frames=_int(
            e, "VIDEO_MAX_FRAMES", 300, min_val=1, max_val=10_000
        ),
        # Group 6: ML paths
        fasttext_model_path=e.get(
            "FASTTEXT_MODEL_PATH", "/app/models/lid.176.bin"
        ),
        source_folder=source_folder,
        output_folder=output_folder,
        temp_folder=temp_folder,
        log_dir=log_dir,
        failure_report=failure_report,
        healthcheck_file=healthcheck_file,
        # Group 7: Document Intelligence (Phase 3A)
        enable_document_intelligence=_bool(
            e, "ENABLE_DOCUMENT_INTELLIGENCE", False
        ),
        enable_layout_analysis=_bool(e, "ENABLE_LAYOUT_ANALYSIS", True),
        enable_table_extraction=_bool(e, "ENABLE_TABLE_EXTRACTION", True),
        docintel_mode=_str_choice(
            e, "DOCINTEL_MODE", "full", _DOCINTEL_MODES
        ),
        export_tables=_bool(e, "EXPORT_TABLES", False),
        # Group 8: Form & KV extraction (Phase 3C)
        enable_form_detection=_bool(e, "ENABLE_FORM_DETECTION", False),
        enable_kv_extraction=_bool(e, "ENABLE_KV_EXTRACTION", False),
        enable_privilege_detection=_bool(
            e, "ENABLE_PRIVILEGE_DETECTION", False
        ),
        # Group 9: Custody
        enable_custody=_bool(e, "ENABLE_CUSTODY", True),
        # Group 10: Processing features
        enable_preprocessing=_bool(e, "ENABLE_PREPROCESSING", False),
        preprocessing_level=_str_choice(
            e, "PREPROCESSING_LEVEL", "standard", _PREPROCESSING_LEVELS
        ),
        enable_noise_profiling=_bool(e, "ENABLE_NOISE_PROFILING", False),
        enable_validation=_bool(e, "ENABLE_VALIDATION", True),
        enable_dpi_escalation=_bool(e, "ENABLE_DPI_ESCALATION", False),
        dpi_confidence_threshold=_float(
            e,
            "DPI_CONFIDENCE_THRESHOLD",
            0.60,
            min_val=0.0,
            max_val=1.0,
        ),
        # Group 11: Analysis features
        enable_ner=_bool(e, "ENABLE_NER", False),
        enable_handwriting=_bool(e, "ENABLE_HANDWRITING", False),
        enable_signature_verification=_bool(
            e, "ENABLE_SIGNATURE_VERIFICATION", False
        ),
        enable_vertical_text=_bool(e, "ENABLE_VERTICAL_TEXT", False),
        enable_table_fallback=_bool(e, "ENABLE_TABLE_FALLBACK", False),
        enable_classification=_bool(e, "ENABLE_CLASSIFICATION", False),
        enable_extraction=_bool(e, "ENABLE_EXTRACTION", False),
        enable_specialist_routing=_bool(
            e, "ENABLE_SPECIALIST_ROUTING", False
        ),
        enable_entity_consolidation=_bool(
            e, "ENABLE_ENTITY_CONSOLIDATION", False
        ),
        enable_relationship_extraction=_bool(
            e, "ENABLE_RELATIONSHIP_EXTRACTION", False
        ),
        enable_retrieval_output=_bool(e, "ENABLE_RETRIEVAL_OUTPUT", False),
        enable_exception_routing=_bool(e, "ENABLE_EXCEPTION_ROUTING", False),
        # Group 12: Pipeline optimization
        enable_adaptive_batch=_bool(e, "ENABLE_ADAPTIVE_BATCH", False),
        enable_page_cache=_bool(e, "ENABLE_PAGE_CACHE", False),
        enable_page_routing=_bool(e, "ENABLE_PAGE_ROUTING", False),
        enable_gpu_optimization=_bool(e, "ENABLE_GPU_OPTIMIZATION", False),
        # Group 13: Per-span language detection (Plan A)
        enable_per_span_language=_bool(e, "ENABLE_PER_SPAN_LANGUAGE", False),
        language_include_spans=_bool(e, "LANGUAGE_INCLUDE_SPANS", False),
        language_short_span_threshold=_int(
            e,
            "LANGUAGE_SHORT_SPAN_THRESHOLD",
            20,
            min_val=1,
            max_val=10_000,
        ),
        language_confidence_threshold=_float(
            e,
            "LANGUAGE_CONFIDENCE_THRESHOLD",
            0.4,
            min_val=0.0,
            max_val=1.0,
        ),
        language_redact_samples=_str_choice(
            e,
            "LANGUAGE_REDACT_SAMPLES",
            "privilege_or_short_doc",
            _LANGUAGE_REDACT_MODES,
        ),
        # Group 14: Translation enrichment (Plan B Wave M1)
        enable_translation=_bool(e, "ENABLE_TRANSLATION", False),
        translation_target_languages=_csv_list(
            e, "TRANSLATION_TARGET_LANGUAGES"
        ),
        translation_quality=_str_choice(
            e,
            "TRANSLATION_QUALITY",
            "standard",
            _TRANSLATION_QUALITY_LEVELS,
        ),
        translation_latency=_str_choice(
            e,
            "TRANSLATION_LATENCY",
            "standard",
            _TRANSLATION_LATENCY_LEVELS,
        ),
        enable_handwriting_mt=_bool(e, "ENABLE_HANDWRITING_MT", False),
        translation_prefer_external_service=_bool(
            e, "EDC_TRANSLATION_PREFER_EXTERNAL", False
        ),
        translation_external_service_url=(
            e.get("EDC_TRANSLATION_URL") if e.get("EDC_TRANSLATION_URL") else None
        ),
        translation_external_provider_id=(
            e.get("EDC_TRANSLATION_PROVIDER_ID")
            if e.get("EDC_TRANSLATION_PROVIDER_ID")
            else "passthrough"
        ),
        translation_external_timeout_seconds=_float(
            e,
            "EDC_TRANSLATION_TIMEOUT_SECONDS",
            30.0,
            min_val=0.1,
            max_val=600.0,
        ),
        translation_external_readiness_path=(
            e.get("EDC_TRANSLATION_READINESS_PATH")
            if e.get("EDC_TRANSLATION_READINESS_PATH")
            else "/health"
        ),
        # Group 15: Translation model cache (Plan B Wave M2)
        translation_cache_dir=(
            e.get("TRANSLATION_MODEL_CACHE_DIR")
            if e.get("TRANSLATION_MODEL_CACHE_DIR")
            else None
        ),
        translation_cache_max_bytes=_int(
            e,
            "TRANSLATION_CACHE_MAX_BYTES",
            50 * 1024 * 1024 * 1024,
            min_val=1,
            max_val=10 * 1024 * 1024 * 1024 * 1024,
        ),
        translation_airgapped=_bool(e, "TRANSLATION_AIRGAPPED", False),
        # Group 16: Batch translation scheduling (Plan B Wave M2 -- B17)
        translation_batch_enabled=_bool(
            e, "OCR_TRANSLATION_BATCH_ENABLED", False
        ),
        translation_batch_max_inputs=_int(
            e,
            "OCR_TRANSLATION_BATCH_MAX_INPUTS",
            1000,
            min_val=1,
            max_val=100_000,
        ),
        translation_batch_input_max_bytes=_int(
            e,
            "OCR_TRANSLATION_BATCH_INPUT_MAX_BYTES",
            8 * 1024,
            min_val=1,
            max_val=10 * 1024 * 1024,
        ),
        translation_batch_fan_out_size=_int(
            e,
            "OCR_TRANSLATION_BATCH_FAN_OUT_SIZE",
            32,
            min_val=1,
            max_val=1024,
        ),
        translation_batch_concurrency=_int(
            e,
            "OCR_TRANSLATION_BATCH_CONCURRENCY",
            4,
            min_val=1,
            max_val=128,
        ),
        # Group 17: Translation model provenance (Plan B Wave M2 -- B19)
        translation_enforce_provenance=_bool(
            e, "OCR_TRANSLATION_ENFORCE_PROVENANCE", False
        ),
    )
