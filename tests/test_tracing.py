"""Tests for api/tracing.py — OpenTelemetry distributed tracing with local fallback."""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest

from api.tracing import (
    LocalSpan,
    LocalTracer,
    SpanKind,
    SpanStatus,
    TracingConfig,
    get_tracer,
    init_tracing,
    trace_operation,
)

# ---------------------------------------------------------------------------
# SpanKind enum
# ---------------------------------------------------------------------------


class TestSpanKind:
    """Tests for SpanKind enum values."""

    def test_enum_has_5_members(self):
        assert len(SpanKind) == 5

    def test_client_value(self):
        assert SpanKind.CLIENT.value == "CLIENT"

    def test_server_value(self):
        assert SpanKind.SERVER.value == "SERVER"

    def test_producer_value(self):
        assert SpanKind.PRODUCER.value == "PRODUCER"

    def test_consumer_value(self):
        assert SpanKind.CONSUMER.value == "CONSUMER"

    def test_internal_value(self):
        assert SpanKind.INTERNAL.value == "INTERNAL"


# ---------------------------------------------------------------------------
# SpanStatus enum
# ---------------------------------------------------------------------------


class TestSpanStatus:
    """Tests for SpanStatus enum values."""

    def test_enum_has_3_members(self):
        assert len(SpanStatus) == 3

    def test_ok_value(self):
        assert SpanStatus.OK.value == "OK"

    def test_error_value(self):
        assert SpanStatus.ERROR.value == "ERROR"

    def test_unset_value(self):
        assert SpanStatus.UNSET.value == "UNSET"


# ---------------------------------------------------------------------------
# LocalSpan creation
# ---------------------------------------------------------------------------


class TestLocalSpanCreation:
    """Tests for LocalSpan construction and auto-populated fields."""

    def test_name_stored(self):
        span = LocalSpan(name="test-op")
        assert span.name == "test-op"

    def test_auto_span_id(self):
        span = LocalSpan(name="op")
        assert span.span_id
        assert len(span.span_id) == 16

    def test_auto_trace_id(self):
        span = LocalSpan(name="op")
        assert span.trace_id
        assert len(span.trace_id) == 32

    def test_unique_span_ids(self):
        ids = {LocalSpan(name="op").span_id for _ in range(50)}
        assert len(ids) == 50

    def test_unique_trace_ids(self):
        ids = {LocalSpan(name="op").trace_id for _ in range(50)}
        assert len(ids) == 50

    def test_default_parent_id_is_none(self):
        span = LocalSpan(name="op")
        assert span.parent_id is None

    def test_default_end_time_is_none(self):
        span = LocalSpan(name="op")
        assert span.end_time is None

    def test_default_attributes_empty_dict(self):
        span = LocalSpan(name="op")
        assert span.attributes == {}

    def test_default_events_empty_list(self):
        span = LocalSpan(name="op")
        assert span.events == []

    def test_default_status_is_unset(self):
        span = LocalSpan(name="op")
        assert span.status == SpanStatus.UNSET

    def test_default_kind_is_internal(self):
        span = LocalSpan(name="op")
        assert span.kind == SpanKind.INTERNAL

    def test_auto_start_time(self):
        before = time.time()
        span = LocalSpan(name="op")
        after = time.time()
        assert before <= span.start_time <= after

    def test_explicit_span_id_preserved(self):
        span = LocalSpan(name="op", span_id="abc123")
        assert span.span_id == "abc123"

    def test_explicit_trace_id_preserved(self):
        span = LocalSpan(name="op", trace_id="trace-xyz")
        assert span.trace_id == "trace-xyz"


# ---------------------------------------------------------------------------
# LocalSpan duration_ms
# ---------------------------------------------------------------------------


class TestLocalSpanDuration:
    """Tests for LocalSpan.duration_ms property."""

    def test_duration_ms_none_when_not_ended(self):
        span = LocalSpan(name="op")
        assert span.duration_ms is None

    def test_duration_ms_calculated(self):
        span = LocalSpan(name="op", start_time=1000.0)
        span.end_time = 1000.05
        assert span.duration_ms == pytest.approx(50.0)

    def test_duration_ms_zero_for_instant(self):
        t = time.time()
        span = LocalSpan(name="op", start_time=t)
        span.end_time = t
        assert span.duration_ms == 0.0


# ---------------------------------------------------------------------------
# LocalSpan to_dict
# ---------------------------------------------------------------------------


class TestLocalSpanToDict:
    """Tests for LocalSpan.to_dict serialisation."""

    def test_to_dict_keys(self):
        span = LocalSpan(name="op")
        d = span.to_dict()
        expected = {
            "name", "span_id", "trace_id", "parent_id", "start_time",
            "end_time", "duration_ms", "attributes", "events", "status", "kind",
        }
        assert set(d.keys()) == expected

    def test_to_dict_name(self):
        span = LocalSpan(name="my-span")
        assert span.to_dict()["name"] == "my-span"

    def test_to_dict_status_is_string(self):
        span = LocalSpan(name="op")
        assert span.to_dict()["status"] == "UNSET"

    def test_to_dict_kind_is_string(self):
        span = LocalSpan(name="op", kind=SpanKind.SERVER)
        assert span.to_dict()["kind"] == "SERVER"

    def test_to_dict_attributes_copy(self):
        span = LocalSpan(name="op", attributes={"k": "v"})
        d = span.to_dict()
        d["attributes"]["k"] = "changed"
        assert span.attributes["k"] == "v"


# ---------------------------------------------------------------------------
# LocalSpan as context manager
# ---------------------------------------------------------------------------


class TestLocalSpanContextManager:
    """Tests for LocalSpan __enter__ / __exit__."""

    def test_context_manager_sets_end_time(self):
        span = LocalSpan(name="op")
        assert span.end_time is None
        with span:
            pass
        assert span.end_time is not None

    def test_context_manager_returns_self(self):
        span = LocalSpan(name="op")
        with span as s:
            assert s is span

    def test_context_manager_sets_ok_on_success(self):
        span = LocalSpan(name="op")
        with span:
            pass
        assert span.status == SpanStatus.OK

    def test_context_manager_sets_error_on_exception(self):
        span = LocalSpan(name="op")
        with pytest.raises(ValueError):
            with span:
                raise ValueError("boom")
        assert span.status == SpanStatus.ERROR

    def test_context_manager_does_not_suppress_exception(self):
        span = LocalSpan(name="op")
        with pytest.raises(RuntimeError, match="oops"):
            with span:
                raise RuntimeError("oops")


# ---------------------------------------------------------------------------
# LocalSpan add_event / set_attribute
# ---------------------------------------------------------------------------


class TestLocalSpanEvents:
    """Tests for add_event and set_attribute helpers."""

    def test_add_event_appends(self):
        span = LocalSpan(name="op")
        span.add_event("checkpoint")
        assert len(span.events) == 1
        assert span.events[0]["name"] == "checkpoint"

    def test_add_event_with_attributes(self):
        span = LocalSpan(name="op")
        span.add_event("done", attributes={"pages": 5})
        assert span.events[0]["attributes"] == {"pages": 5}

    def test_add_event_has_timestamp(self):
        before = time.time()
        span = LocalSpan(name="op")
        span.add_event("tick")
        after = time.time()
        assert before <= span.events[0]["timestamp"] <= after

    def test_set_attribute(self):
        span = LocalSpan(name="op")
        span.set_attribute("file", "doc.pdf")
        assert span.attributes["file"] == "doc.pdf"

    def test_set_attribute_overwrites(self):
        span = LocalSpan(name="op")
        span.set_attribute("k", 1)
        span.set_attribute("k", 2)
        assert span.attributes["k"] == 2


# ---------------------------------------------------------------------------
# LocalTracer construction
# ---------------------------------------------------------------------------


class TestLocalTracerConstruction:
    """Tests for LocalTracer initialisation."""

    def test_default_service_name(self):
        tracer = LocalTracer()
        assert tracer.service_name == "ocr-local-api"

    def test_custom_service_name(self):
        tracer = LocalTracer(service_name="my-service")
        assert tracer.service_name == "my-service"

    def test_starts_with_no_spans(self):
        tracer = LocalTracer()
        assert tracer.get_spans() == []


# ---------------------------------------------------------------------------
# LocalTracer start_span
# ---------------------------------------------------------------------------


class TestLocalTracerStartSpan:
    """Tests for LocalTracer.start_span."""

    def test_returns_local_span(self):
        tracer = LocalTracer()
        span = tracer.start_span("op")
        assert isinstance(span, LocalSpan)

    def test_span_name_set(self):
        tracer = LocalTracer()
        span = tracer.start_span("my-operation")
        assert span.name == "my-operation"

    def test_span_kind_set(self):
        tracer = LocalTracer()
        span = tracer.start_span("op", kind=SpanKind.CLIENT)
        assert span.kind == SpanKind.CLIENT

    def test_span_attributes_set(self):
        tracer = LocalTracer()
        span = tracer.start_span("op", attributes={"x": 42})
        assert span.attributes == {"x": 42}

    def test_span_recorded(self):
        tracer = LocalTracer()
        tracer.start_span("a")
        tracer.start_span("b")
        assert len(tracer.get_spans()) == 2


# ---------------------------------------------------------------------------
# LocalTracer start_span with parent
# ---------------------------------------------------------------------------


class TestLocalTracerParent:
    """Tests for parent span propagation."""

    def test_parent_id_set(self):
        tracer = LocalTracer()
        parent = tracer.start_span("parent")
        child = tracer.start_span("child", parent=parent)
        assert child.parent_id == parent.span_id

    def test_trace_id_inherited(self):
        tracer = LocalTracer()
        parent = tracer.start_span("parent")
        child = tracer.start_span("child", parent=parent)
        assert child.trace_id == parent.trace_id

    def test_no_parent_gives_none_parent_id(self):
        tracer = LocalTracer()
        span = tracer.start_span("root")
        assert span.parent_id is None


# ---------------------------------------------------------------------------
# LocalTracer get_spans
# ---------------------------------------------------------------------------


class TestLocalTracerGetSpans:
    """Tests for get_spans with and without limit."""

    def test_get_spans_returns_all(self):
        tracer = LocalTracer()
        for i in range(10):
            tracer.start_span(f"op-{i}")
        assert len(tracer.get_spans()) == 10

    def test_get_spans_with_limit(self):
        tracer = LocalTracer()
        for i in range(10):
            tracer.start_span(f"op-{i}")
        spans = tracer.get_spans(limit=3)
        assert len(spans) == 3

    def test_get_spans_limit_returns_most_recent(self):
        tracer = LocalTracer()
        for i in range(10):
            tracer.start_span(f"op-{i}")
        spans = tracer.get_spans(limit=2)
        assert [s.name for s in spans] == ["op-8", "op-9"]

    def test_get_spans_returns_copy(self):
        tracer = LocalTracer()
        tracer.start_span("op")
        spans = tracer.get_spans()
        spans.clear()
        assert len(tracer.get_spans()) == 1


# ---------------------------------------------------------------------------
# LocalTracer clear
# ---------------------------------------------------------------------------


class TestLocalTracerClear:
    """Tests for LocalTracer.clear."""

    def test_clear_removes_spans(self):
        tracer = LocalTracer()
        tracer.start_span("a")
        tracer.start_span("b")
        tracer.clear()
        assert tracer.get_spans() == []

    def test_clear_on_empty_is_noop(self):
        tracer = LocalTracer()
        tracer.clear()
        assert tracer.get_spans() == []


# ---------------------------------------------------------------------------
# LocalTracer thread safety
# ---------------------------------------------------------------------------


class TestLocalTracerThreadSafety:
    """Tests for concurrent LocalTracer operations."""

    def test_concurrent_start_span(self):
        tracer = LocalTracer()
        count = 100
        barrier = threading.Barrier(4)

        def worker():
            barrier.wait()
            for i in range(count):
                tracer.start_span(f"op-{i}")

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(tracer.get_spans()) == count * 4

    def test_concurrent_get_spans_and_start_span(self):
        tracer = LocalTracer()

        def writer():
            for _ in range(100):
                tracer.start_span("w")

        def reader():
            for _ in range(100):
                tracer.get_spans()

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        # no crash — pass


# ---------------------------------------------------------------------------
# TracingConfig defaults
# ---------------------------------------------------------------------------


class TestTracingConfigDefaults:
    """Tests for TracingConfig default values."""

    def test_enabled_default_true(self):
        cfg = TracingConfig()
        assert cfg.enabled is True

    def test_service_name_default(self):
        cfg = TracingConfig()
        assert cfg.service_name == "ocr-local-api"

    def test_exporter_default(self):
        cfg = TracingConfig()
        assert cfg.exporter == "console"

    def test_endpoint_default(self):
        cfg = TracingConfig()
        assert cfg.endpoint == "http://localhost:4317"

    def test_sample_rate_default(self):
        cfg = TracingConfig()
        assert cfg.sample_rate == 1.0


# ---------------------------------------------------------------------------
# TracingConfig from_env
# ---------------------------------------------------------------------------


class TestTracingConfigFromEnv:
    """Tests for TracingConfig.from_env reading OTEL_* env vars."""

    def test_from_env_reads_enabled(self):
        with patch.dict("os.environ", {"OTEL_ENABLED": "false"}):
            cfg = TracingConfig.from_env()
        assert cfg.enabled is False

    def test_from_env_reads_service_name(self):
        with patch.dict("os.environ", {"OTEL_SERVICE_NAME": "my-svc"}):
            cfg = TracingConfig.from_env()
        assert cfg.service_name == "my-svc"

    def test_from_env_reads_exporter(self):
        with patch.dict("os.environ", {"OTEL_EXPORTER": "otlp"}):
            cfg = TracingConfig.from_env()
        assert cfg.exporter == "otlp"

    def test_from_env_reads_endpoint(self):
        with patch.dict("os.environ", {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://otel:4318"}):
            cfg = TracingConfig.from_env()
        assert cfg.endpoint == "http://otel:4318"

    def test_from_env_reads_sample_rate(self):
        with patch.dict("os.environ", {"OTEL_SAMPLE_RATE": "0.5"}):
            cfg = TracingConfig.from_env()
        assert cfg.sample_rate == 0.5

    def test_from_env_defaults_when_vars_absent(self):
        env = {
            k: v for k, v in __import__("os").environ.items()
            if not k.startswith("OTEL_")
        }
        with patch.dict("os.environ", env, clear=True):
            cfg = TracingConfig.from_env()
        assert cfg.enabled is True
        assert cfg.service_name == "ocr-local-api"
        assert cfg.exporter == "console"
        assert cfg.endpoint == "http://localhost:4317"
        assert cfg.sample_rate == 1.0


# ---------------------------------------------------------------------------
# init_tracing
# ---------------------------------------------------------------------------


class TestInitTracing:
    """Tests for init_tracing fallback behaviour."""

    def test_returns_local_tracer_when_otel_unavailable(self):
        import api.tracing as tracing_mod

        tracing_mod._global_tracer = None
        with patch.dict("sys.modules", {"opentelemetry": None}):
            tracer = init_tracing()
        assert isinstance(tracer, LocalTracer)

    def test_returns_local_tracer_when_disabled(self):
        import api.tracing as tracing_mod

        tracing_mod._global_tracer = None
        cfg = TracingConfig(enabled=False)
        tracer = init_tracing(cfg)
        assert isinstance(tracer, LocalTracer)

    def test_disabled_config_uses_service_name(self):
        import api.tracing as tracing_mod

        tracing_mod._global_tracer = None
        cfg = TracingConfig(enabled=False, service_name="custom-svc")
        tracer = init_tracing(cfg)
        assert isinstance(tracer, LocalTracer)
        assert tracer.service_name == "custom-svc"


# ---------------------------------------------------------------------------
# get_tracer
# ---------------------------------------------------------------------------


class TestGetTracer:
    """Tests for get_tracer global accessor."""

    def test_returns_local_tracer_by_default(self):
        import api.tracing as tracing_mod

        tracing_mod._global_tracer = None
        tracer = get_tracer()
        assert isinstance(tracer, LocalTracer)

    def test_returns_same_instance_on_repeated_calls(self):
        import api.tracing as tracing_mod

        tracing_mod._global_tracer = None
        t1 = get_tracer()
        t2 = get_tracer()
        assert t1 is t2


# ---------------------------------------------------------------------------
# trace_operation
# ---------------------------------------------------------------------------


class TestTraceOperation:
    """Tests for the trace_operation context manager."""

    def test_context_manager_yields_span(self):
        import api.tracing as tracing_mod

        tracing_mod._global_tracer = LocalTracer()
        with trace_operation("my-op") as span:
            assert isinstance(span, LocalSpan)
            assert span.name == "my-op"

    def test_records_span_in_tracer(self):
        import api.tracing as tracing_mod

        tracer = LocalTracer()
        tracing_mod._global_tracer = tracer
        with trace_operation("op"):
            pass
        spans = tracer.get_spans()
        assert len(spans) == 1
        assert spans[0].name == "op"

    def test_records_attributes(self):
        import api.tracing as tracing_mod

        tracer = LocalTracer()
        tracing_mod._global_tracer = tracer
        with trace_operation("op", attributes={"page": 7, "engine": "paddle"}):
            pass
        span = tracer.get_spans()[0]
        assert span.attributes["page"] == 7
        assert span.attributes["engine"] == "paddle"

    def test_sets_ok_on_success(self):
        import api.tracing as tracing_mod

        tracer = LocalTracer()
        tracing_mod._global_tracer = tracer
        with trace_operation("op"):
            pass
        span = tracer.get_spans()[0]
        assert span.status == SpanStatus.OK

    def test_sets_error_on_exception(self):
        import api.tracing as tracing_mod

        tracer = LocalTracer()
        tracing_mod._global_tracer = tracer
        with pytest.raises(RuntimeError):
            with trace_operation("op"):
                raise RuntimeError("fail")
        span = tracer.get_spans()[0]
        assert span.status == SpanStatus.ERROR

    def test_exception_propagates(self):
        import api.tracing as tracing_mod

        tracing_mod._global_tracer = LocalTracer()
        with pytest.raises(ValueError, match="test-err"):
            with trace_operation("op"):
                raise ValueError("test-err")

    def test_span_end_time_set_after_block(self):
        import api.tracing as tracing_mod

        tracer = LocalTracer()
        tracing_mod._global_tracer = tracer
        with trace_operation("op"):
            pass
        span = tracer.get_spans()[0]
        assert span.end_time is not None
        assert span.duration_ms is not None
        assert span.duration_ms >= 0

    def test_kind_passed_to_span(self):
        import api.tracing as tracing_mod

        tracer = LocalTracer()
        tracing_mod._global_tracer = tracer
        with trace_operation("op", kind=SpanKind.SERVER):
            pass
        span = tracer.get_spans()[0]
        assert span.kind == SpanKind.SERVER
