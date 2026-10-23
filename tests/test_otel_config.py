"""Tests for OpenTelemetry tracing configuration behavior.

Covers TracingConfig sampling strategy, sample_rate clamping, from_env,
and init_tracing. Collector YAML and compose file text are not pinned.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from api.tracing import (
    LocalTracer,
    TracingConfig,
    init_tracing,
)



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


