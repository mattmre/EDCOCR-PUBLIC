"""Tests for OpenTelemetry production collector configuration.

Covers:
- TracingConfig production fields (sampling_strategy, sample_rate clamping)
- from_env with OTEL_SAMPLING_STRATEGY and OTEL_SAMPLING_RATE
- Collector config YAML validity
- Docker Compose overlay structure
- init_tracing with new sampling strategies
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from api.tracing import (
    LocalTracer,
    TracingConfig,
    init_tracing,
)

# ---------------------------------------------------------------------------
# Locate the otel/ directory relative to repo root
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
OTEL_DIR = REPO_ROOT / "otel"
COLLECTOR_CONFIG = OTEL_DIR / "otel-collector-config.yaml"
COMPOSE_FILE = OTEL_DIR / "docker-compose.otel.yml"


# ---------------------------------------------------------------------------
# Helper: load YAML safely (stdlib fallback if pyyaml unavailable)
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    """Load a YAML file, returning a dict.  Falls back to raw text check."""
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except ImportError:
        # No pyyaml — do basic structural checks via text
        return {"_raw": path.read_text(encoding="utf-8")}


# ===========================================================================
# TracingConfig production defaults
# ===========================================================================


class TestTracingConfigSamplingStrategy:
    """Tests for the sampling_strategy field on TracingConfig."""

    def test_default_sampling_strategy_is_parentbased(self):
        cfg = TracingConfig()
        assert cfg.sampling_strategy == "parentbased"

    def test_parentbased_strategy_preserved(self):
        cfg = TracingConfig(sampling_strategy="parentbased")
        assert cfg.sampling_strategy == "parentbased"

    def test_invalid_strategy_falls_back_to_parentbased(self):
        cfg = TracingConfig(sampling_strategy="unknown")
        assert cfg.sampling_strategy == "parentbased"

    def test_empty_string_strategy_falls_back_to_parentbased(self):
        cfg = TracingConfig(sampling_strategy="")
        assert cfg.sampling_strategy == "parentbased"


class TestTracingConfigSampleRateClamping:
    """Tests for sample_rate boundary clamping in __post_init__."""

    def test_rate_clamped_to_zero_floor(self):
        cfg = TracingConfig(sample_rate=-0.5)
        assert cfg.sample_rate == 0.0

    def test_rate_clamped_to_one_ceiling(self):
        cfg = TracingConfig(sample_rate=2.0)
        assert cfg.sample_rate == 1.0

    def test_rate_zero_preserved(self):
        cfg = TracingConfig(sample_rate=0.0)
        assert cfg.sample_rate == 0.0

    def test_rate_one_preserved(self):
        cfg = TracingConfig(sample_rate=1.0)
        assert cfg.sample_rate == 1.0

    def test_rate_mid_value_preserved(self):
        cfg = TracingConfig(sample_rate=0.1)
        assert cfg.sample_rate == pytest.approx(0.1)

    def test_rate_tiny_value_preserved(self):
        cfg = TracingConfig(sample_rate=0.001)
        assert cfg.sample_rate == pytest.approx(0.001)


# ===========================================================================
# TracingConfig.from_env with new fields
# ===========================================================================


class TestTracingConfigFromEnvSampling:
    """Tests for from_env reading OTEL_SAMPLING_* env vars."""

    def test_from_env_reads_sampling_strategy(self):
        with patch.dict("os.environ", {"OTEL_SAMPLING_STRATEGY": "parentbased"}):
            cfg = TracingConfig.from_env()
        assert cfg.sampling_strategy == "parentbased"

    def test_from_env_default_sampling_strategy(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("OTEL_")}
        with patch.dict("os.environ", env, clear=True):
            cfg = TracingConfig.from_env()
        assert cfg.sampling_strategy == "parentbased"

    def test_from_env_invalid_sampling_rate_string(self):
        with patch.dict("os.environ", {"OTEL_SAMPLE_RATE": "not-a-number"}):
            cfg = TracingConfig.from_env()
        assert cfg.sample_rate == 1.0

    def test_from_env_negative_rate_clamped(self):
        with patch.dict("os.environ", {"OTEL_SAMPLE_RATE": "-5.0"}):
            cfg = TracingConfig.from_env()
        assert cfg.sample_rate == 0.0

    def test_from_env_excessive_rate_clamped(self):
        with patch.dict("os.environ", {"OTEL_SAMPLE_RATE": "99.0"}):
            cfg = TracingConfig.from_env()
        assert cfg.sample_rate == 1.0

    def test_from_env_production_rate(self):
        with patch.dict("os.environ", {"OTEL_SAMPLE_RATE": "0.1"}):
            cfg = TracingConfig.from_env()
        assert cfg.sample_rate == pytest.approx(0.1)

    def test_from_env_reads_all_new_fields(self):
        env_overrides = {
            "OTEL_SAMPLE_RATE": "0.05",
            "OTEL_SAMPLING_STRATEGY": "parentbased",
            "OTEL_SERVICE_NAME": "ocr-local-api",
        }
        with patch.dict("os.environ", env_overrides):
            cfg = TracingConfig.from_env()
        assert cfg.sample_rate == pytest.approx(0.05)
        assert cfg.sampling_strategy == "parentbased"
        assert cfg.service_name == "ocr-local-api"


# ===========================================================================
# init_tracing with sampling strategies
# ===========================================================================


class TestInitTracingSampling:
    """Tests for init_tracing using the new sampling_strategy field."""

    def test_disabled_config_ignores_strategy(self):
        import api.tracing as tracing_mod
        tracing_mod._global_tracer = None
        cfg = TracingConfig(enabled=False, sampling_strategy="parentbased")
        tracer = init_tracing(cfg)
        assert isinstance(tracer, LocalTracer)

    def test_local_tracer_fallback_with_parentbased(self):
        """When OTel SDK is missing, strategy is irrelevant — LocalTracer used."""
        import api.tracing as tracing_mod
        tracing_mod._global_tracer = None
        with patch.dict("sys.modules", {"opentelemetry": None}):
            cfg = TracingConfig(
                enabled=True,
                sampling_strategy="parentbased",
                sample_rate=0.1,
            )
            tracer = init_tracing(cfg)
        assert isinstance(tracer, LocalTracer)

    def test_default_config_produces_tracer(self):
        import api.tracing as tracing_mod
        tracing_mod._global_tracer = None
        tracer = init_tracing(TracingConfig())
        # Without OTel SDK this falls back to LocalTracer; either is acceptable
        assert tracer is not None


# ===========================================================================
# Collector config YAML validity
# ===========================================================================


class TestCollectorConfigYaml:
    """Tests that otel/otel-collector-config.yaml is well-formed."""

    def test_file_exists(self):
        assert COLLECTOR_CONFIG.exists(), f"{COLLECTOR_CONFIG} not found"

    def test_yaml_parses(self):
        data = _load_yaml(COLLECTOR_CONFIG)
        assert data is not None

    def test_has_receivers_section(self):
        data = _load_yaml(COLLECTOR_CONFIG)
        if "_raw" in data:
            assert "receivers:" in data["_raw"]
        else:
            assert "receivers" in data

    def test_has_processors_section(self):
        data = _load_yaml(COLLECTOR_CONFIG)
        if "_raw" in data:
            assert "processors:" in data["_raw"]
        else:
            assert "processors" in data

    def test_has_exporters_section(self):
        data = _load_yaml(COLLECTOR_CONFIG)
        if "_raw" in data:
            assert "exporters:" in data["_raw"]
        else:
            assert "exporters" in data

    def test_has_service_section(self):
        data = _load_yaml(COLLECTOR_CONFIG)
        if "_raw" in data:
            assert "service:" in data["_raw"]
        else:
            assert "service" in data

    def test_has_extensions_section(self):
        data = _load_yaml(COLLECTOR_CONFIG)
        if "_raw" in data:
            assert "extensions:" in data["_raw"]
        else:
            assert "extensions" in data

    def test_otlp_grpc_receiver_configured(self):
        raw = COLLECTOR_CONFIG.read_text(encoding="utf-8")
        assert "4317" in raw

    def test_otlp_http_receiver_configured(self):
        raw = COLLECTOR_CONFIG.read_text(encoding="utf-8")
        assert "4318" in raw

    def test_health_check_extension_on_13133(self):
        raw = COLLECTOR_CONFIG.read_text(encoding="utf-8")
        assert "13133" in raw

    def test_batch_processor_present(self):
        raw = COLLECTOR_CONFIG.read_text(encoding="utf-8")
        assert "batch" in raw

    def test_memory_limiter_present(self):
        raw = COLLECTOR_CONFIG.read_text(encoding="utf-8")
        assert "memory_limiter" in raw

    def test_prometheus_exporter_present(self):
        raw = COLLECTOR_CONFIG.read_text(encoding="utf-8")
        assert "prometheus" in raw

    def test_traces_pipeline_present(self):
        raw = COLLECTOR_CONFIG.read_text(encoding="utf-8")
        assert "traces:" in raw

    def test_metrics_pipeline_present(self):
        raw = COLLECTOR_CONFIG.read_text(encoding="utf-8")
        assert "metrics:" in raw

    def test_logs_pipeline_present(self):
        raw = COLLECTOR_CONFIG.read_text(encoding="utf-8")
        assert "logs:" in raw

    def test_attributes_processor_adds_namespace(self):
        raw = COLLECTOR_CONFIG.read_text(encoding="utf-8")
        assert "ocr-local" in raw

    def test_resource_processor_adds_version(self):
        from version import __version__

        raw = COLLECTOR_CONFIG.read_text(encoding="utf-8")
        assert __version__ in raw


# ===========================================================================
# Docker Compose overlay validation
# ===========================================================================


class TestDockerComposeOtel:
    """Tests that otel/docker-compose.otel.yml is well-formed."""

    def test_file_exists(self):
        assert COMPOSE_FILE.exists(), f"{COMPOSE_FILE} not found"

    def test_yaml_parses(self):
        data = _load_yaml(COMPOSE_FILE)
        assert data is not None

    def test_has_services_section(self):
        data = _load_yaml(COMPOSE_FILE)
        if "_raw" in data:
            assert "services:" in data["_raw"]
        else:
            assert "services" in data

    def test_otel_collector_service_defined(self):
        raw = COMPOSE_FILE.read_text(encoding="utf-8")
        assert "otel-collector:" in raw

    def test_jaeger_service_defined(self):
        raw = COMPOSE_FILE.read_text(encoding="utf-8")
        assert "jaeger:" in raw

    def test_jaeger_ui_port_16686(self):
        raw = COMPOSE_FILE.read_text(encoding="utf-8")
        assert "16686" in raw

    def test_otlp_grpc_port_4317(self):
        raw = COMPOSE_FILE.read_text(encoding="utf-8")
        assert "4317" in raw

    def test_otlp_http_port_4318(self):
        raw = COMPOSE_FILE.read_text(encoding="utf-8")
        assert "4318" in raw

    def test_health_check_port_13133(self):
        raw = COMPOSE_FILE.read_text(encoding="utf-8")
        assert "13133" in raw

    def test_collector_uses_contrib_image(self):
        raw = COMPOSE_FILE.read_text(encoding="utf-8")
        assert "opentelemetry-collector-contrib" in raw

    def test_jaeger_uses_all_in_one_image(self):
        raw = COMPOSE_FILE.read_text(encoding="utf-8")
        assert "jaegertracing/all-in-one" in raw

    def test_collector_mounts_config(self):
        raw = COMPOSE_FILE.read_text(encoding="utf-8")
        assert "otel-collector-config.yaml" in raw


# ===========================================================================
# README existence
# ===========================================================================


class TestOtelReadme:
    """Tests that otel/README.md exists and has expected content."""

    def test_readme_exists(self):
        readme = OTEL_DIR / "README.md"
        assert readme.exists()

    def test_readme_mentions_quick_start(self):
        readme = OTEL_DIR / "README.md"
        text = readme.read_text(encoding="utf-8")
        assert "Quick Start" in text

    def test_readme_mentions_production(self):
        readme = OTEL_DIR / "README.md"
        text = readme.read_text(encoding="utf-8")
        assert "Production" in text

    def test_readme_mentions_sampling(self):
        readme = OTEL_DIR / "README.md"
        text = readme.read_text(encoding="utf-8")
        assert "Sampling" in text

    def test_readme_mentions_environment_variables(self):
        readme = OTEL_DIR / "README.md"
        text = readme.read_text(encoding="utf-8")
        assert "Environment Variable" in text or "OTEL_" in text
