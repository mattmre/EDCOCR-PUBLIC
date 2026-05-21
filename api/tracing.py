"""OpenTelemetry distributed tracing for the OCR pipeline.

Provides trace/span instrumentation with a local fallback when the
OpenTelemetry SDK is not installed.  All ``opentelemetry`` imports are
lazy (try/except inside functions) so the module works without OTel
dependencies -- tests and lightweight deployments use the in-memory
:class:`LocalTracer` / :class:`LocalSpan` path instead.

Production configuration supports standard OTel environment variables:

- **OTEL_EXPORTER_OTLP_ENDPOINT**: OTLP gRPC endpoint
  (default ``http://localhost:4317``).
- **OTEL_EXPORTER_OTLP_HEADERS**: Comma-separated ``key=value`` pairs for
  exporter auth headers (e.g. Grafana Cloud bearer tokens).
- **OTEL_TRACES_SAMPLER**: Sampler type -- ``always_on``, ``always_off``,
  or ``parentbased_traceidratio`` (default ``parentbased_traceidratio``).
- **OTEL_TRACES_SAMPLER_ARG**: Sampling ratio 0.0-1.0 (default ``1.0``).
- **OTEL_SERVICE_NAME**: Service identity (default ``ocr-local-api``).
- **OTEL_SERVICE_VERSION**: Version string (default: ``version.__version__``).
- **OTEL_ENVIRONMENT**: Deployment environment label
  (default: ``DEPLOYMENT_ENV`` or ``development``).

Legacy environment variables (``OTEL_SAMPLE_RATE``, ``OTEL_SAMPLING_STRATEGY``,
``OTEL_EXPORTER``) are still honoured for backward compatibility but the
standard OTel names above take precedence.

Typical usage::

    from api.tracing import init_tracing, get_tracer, trace_operation

    init_tracing()                       # noop if OTel unavailable
    tracer = get_tracer()

    with trace_operation("process_page", attributes={"page": 1}):
        ...

FastAPI integration::

    from api.tracing import configure_tracing

    app = FastAPI()
    configure_tracing(app)               # init + auto-instrument
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Version import (best-effort)
# ---------------------------------------------------------------------------

_SERVICE_VERSION_DEFAULT = "4.1.0"
try:
    from ocr_local.config.version import __version__ as _service_version
except ImportError:
    _service_version = _SERVICE_VERSION_DEFAULT


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SpanKind(Enum):
    """Mirror of ``opentelemetry.trace.SpanKind``."""

    CLIENT = "CLIENT"
    SERVER = "SERVER"
    PRODUCER = "PRODUCER"
    CONSUMER = "CONSUMER"
    INTERNAL = "INTERNAL"


class SpanStatus(Enum):
    """Mirror of ``opentelemetry.trace.StatusCode``."""

    OK = "OK"
    ERROR = "ERROR"
    UNSET = "UNSET"


# ---------------------------------------------------------------------------
# Local (in-memory) span
# ---------------------------------------------------------------------------


@dataclass
class LocalSpan:
    """Lightweight span that records timing and metadata in memory.

    Supports the context-manager protocol so it can be used with ``with``:

        with tracer.start_span("work") as span:
            span.set_attribute("key", "value")
    """

    name: str
    span_id: str = ""
    trace_id: str = ""
    parent_id: Optional[str] = None
    start_time: float = 0.0
    end_time: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    status: SpanStatus = SpanStatus.UNSET
    kind: SpanKind = SpanKind.INTERNAL

    def __post_init__(self):
        if not self.span_id:
            self.span_id = uuid.uuid4().hex[:16]
        if not self.trace_id:
            self.trace_id = uuid.uuid4().hex
        if not self.start_time:
            self.start_time = time.time()

    # -- helpers -------------------------------------------------------------

    @property
    def duration_ms(self) -> Optional[float]:
        """Elapsed wall-clock time in milliseconds, or *None* if not ended."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000.0

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })

    def set_status(self, status: SpanStatus, description: str = "") -> None:
        self.status = status
        if description:
            self.set_attribute("status_description", description)

    def end(self) -> None:
        if self.end_time is None:
            self.end_time = time.time()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "attributes": dict(self.attributes),
            "events": list(self.events),
            "status": self.status.value,
            "kind": self.kind.value,
        }

    # -- context manager -----------------------------------------------------

    def __enter__(self) -> "LocalSpan":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            self.set_status(SpanStatus.ERROR, str(exc_val))
        elif self.status == SpanStatus.UNSET:
            self.set_status(SpanStatus.OK)
        self.end()
        return None  # do not suppress exceptions


# ---------------------------------------------------------------------------
# Local (in-memory) tracer
# ---------------------------------------------------------------------------


class LocalTracer:
    """Thread-safe in-memory tracer that creates :class:`LocalSpan` instances.

    Used as the fallback when OpenTelemetry is not installed.
    """

    def __init__(self, service_name: str = "ocr-local-api"):
        self.service_name = service_name
        self._lock = threading.Lock()
        self._spans: List[LocalSpan] = []

    def start_span(
        self,
        name: str,
        *,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
        parent: Optional[LocalSpan] = None,
    ) -> LocalSpan:
        """Create and register a new :class:`LocalSpan`."""
        parent_id = parent.span_id if parent is not None else None
        trace_id = parent.trace_id if parent is not None else ""
        span = LocalSpan(
            name=name,
            kind=kind,
            attributes=attributes or {},
            parent_id=parent_id,
            trace_id=trace_id,
        )
        with self._lock:
            self._spans.append(span)
        return span

    def get_spans(self, limit: Optional[int] = None) -> List[LocalSpan]:
        """Return recorded spans (most-recent-last).

        If *limit* is given, only the last *limit* spans are returned.
        """
        with self._lock:
            if limit is not None:
                return list(self._spans[-limit:])
            return list(self._spans)

    def clear(self) -> None:
        """Remove all recorded spans."""
        with self._lock:
            self._spans.clear()


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------


def _parse_otlp_headers(raw: str) -> Dict[str, str]:
    """Parse ``OTEL_EXPORTER_OTLP_HEADERS`` value into a dict.

    The spec defines the format as comma-separated ``key=value`` pairs.
    Whitespace around keys and values is stripped.  Empty entries are ignored.

    Examples::

        "Authorization=Bearer tok123,x-scope-orgid=12345"
        => {"Authorization": "Bearer tok123", "x-scope-orgid": "12345"}
    """
    headers: Dict[str, str] = {}
    if not raw or not raw.strip():
        return headers
    for pair in raw.split(","):
        pair = pair.strip()
        if "=" not in pair:
            continue
        key, _, value = pair.partition("=")
        key = key.strip()
        value = value.strip()
        if key:
            headers[key] = value
    return headers


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Valid sampler names (OTel spec subset)
_VALID_SAMPLERS = {"always_on", "always_off", "parentbased_traceidratio"}

# Legacy sampler mapping
_LEGACY_STRATEGY_MAP = {
    "ratio": "always_on",  # legacy "ratio" becomes trace-id-ratio
    "parentbased": "parentbased_traceidratio",
}


@dataclass
class TracingConfig:
    """Configuration for the tracing subsystem.

    All settings can be overridden via ``OTEL_*`` environment variables
    through the :meth:`from_env` class method.

    Attributes:
        enabled: Master toggle for tracing.
        service_name: Service identity reported in spans.
        service_version: Service version string for resource attributes.
        environment: Deployment environment label (``development``,
            ``staging``, ``production``).
        exporter: Backend type -- ``"console"``, ``"otlp"``, or ``"jaeger"``.
        endpoint: OTLP gRPC endpoint (used when exporter is ``"otlp"``).
        headers: Dict of headers to send with OTLP export requests
            (e.g. auth tokens for Grafana Cloud).
        sample_rate: Float 0.0-1.0 controlling what fraction of traces
            are sampled.  Default 1.0 (all traces) is suitable for
            development; production deployments should use 0.1 or lower.
        sampler: OTel sampler name -- ``"always_on"``, ``"always_off"``,
            or ``"parentbased_traceidratio"`` (default).
        sampling_strategy: **Deprecated** legacy alias. Prefer *sampler*.
    """

    enabled: bool = True
    service_name: str = "ocr-local-api"
    service_version: str = ""
    environment: str = "development"
    exporter: str = "console"  # "console" | "otlp" | "jaeger"
    endpoint: str = "http://localhost:4317"
    headers: Dict[str, str] = field(default_factory=dict)
    sample_rate: float = 1.0
    sampler: str = "parentbased_traceidratio"
    sampling_strategy: str = "parentbased"  # legacy compat

    def __post_init__(self):
        """Validate and clamp configuration values."""
        self.sample_rate = max(0.0, min(1.0, self.sample_rate))
        if self.sampler not in _VALID_SAMPLERS:
            self.sampler = "parentbased_traceidratio"
        if self.sampling_strategy not in ("ratio", "parentbased"):
            self.sampling_strategy = "parentbased"
        if not self.service_version:
            self.service_version = _service_version

    @classmethod
    def from_env(cls) -> "TracingConfig":
        """Build a :class:`TracingConfig` from environment variables.

        Reads **standard** OTel env vars first, with legacy fallbacks:

        +-----------------------------------+-------------------------------------+
        | Standard env var                  | Legacy fallback                     |
        +===================================+=====================================+
        | ``OTEL_SERVICE_NAME``             | (default ``ocr-local-api``)         |
        | ``OTEL_SERVICE_VERSION``          | ``version.__version__``             |
        | ``OTEL_EXPORTER_OTLP_ENDPOINT``   | (default ``http://localhost:4317``) |
        | ``OTEL_EXPORTER_OTLP_HEADERS``    | (none)                              |
        | ``OTEL_TRACES_SAMPLER``           | ``OTEL_SAMPLING_STRATEGY``          |
        | ``OTEL_TRACES_SAMPLER_ARG``       | ``OTEL_SAMPLE_RATE``                |
        | ``OTEL_ENVIRONMENT``              | ``DEPLOYMENT_ENV``                  |
        | ``OTEL_ENABLED``                  | (default ``true``)                  |
        | ``OTEL_EXPORTER``                 | (default ``console``)               |
        +-----------------------------------+-------------------------------------+
        """
        # --- enabled ---
        enabled = os.environ.get("OTEL_ENABLED", "true").lower() == "true"

        # --- service identity ---
        service_name = os.environ.get(
            "OTEL_SERVICE_NAME", "ocr-local-api",
        )
        service_version = os.environ.get(
            "OTEL_SERVICE_VERSION", _service_version,
        )

        # --- environment ---
        environment = (
            os.environ.get("OTEL_ENVIRONMENT")
            or os.environ.get("DEPLOYMENT_ENV")
            or "development"
        )

        # --- exporter ---
        exporter = os.environ.get("OTEL_EXPORTER", "console")
        endpoint = os.environ.get(
            "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317",
        )
        headers = _parse_otlp_headers(
            os.environ.get("OTEL_EXPORTER_OTLP_HEADERS", ""),
        )

        # --- sampling (standard takes precedence over legacy) ---
        raw_sampler = os.environ.get("OTEL_TRACES_SAMPLER", "")
        legacy_strategy = os.environ.get("OTEL_SAMPLING_STRATEGY", "")

        if raw_sampler and raw_sampler in _VALID_SAMPLERS:
            sampler = raw_sampler
        elif legacy_strategy:
            sampler = _LEGACY_STRATEGY_MAP.get(
                legacy_strategy, "parentbased_traceidratio",
            )
        else:
            sampler = "parentbased_traceidratio"

        # Map sampler back to legacy strategy for backward compat
        if sampler == "parentbased_traceidratio":
            sampling_strategy = "parentbased"
        else:
            sampling_strategy = "ratio"

        # Sampling ratio: standard OTEL_TRACES_SAMPLER_ARG > legacy OTEL_SAMPLE_RATE
        raw_rate = os.environ.get(
            "OTEL_TRACES_SAMPLER_ARG",
            os.environ.get("OTEL_SAMPLE_RATE", "1.0"),
        )
        try:
            rate = float(raw_rate)
        except (ValueError, TypeError):
            rate = 1.0

        return cls(
            enabled=enabled,
            service_name=service_name,
            service_version=service_version,
            environment=environment,
            exporter=exporter,
            endpoint=endpoint,
            headers=headers,
            sample_rate=rate,
            sampler=sampler,
            sampling_strategy=sampling_strategy,
        )


# ---------------------------------------------------------------------------
# Global tracer state
# ---------------------------------------------------------------------------

_global_tracer: Optional[Any] = None
_tracer_lock = threading.Lock()


def init_tracing(config: Optional[TracingConfig] = None) -> Any:
    """Initialise the tracing subsystem.

    Attempts to configure the OpenTelemetry SDK.  If the SDK packages are
    not installed, or the configuration has ``enabled=False``, a
    :class:`LocalTracer` is used instead.

    Returns the active tracer instance.
    """
    global _global_tracer

    if config is None:
        config = TracingConfig.from_env()

    if not config.enabled:
        tracer = LocalTracer(service_name=config.service_name)
        with _tracer_lock:
            _global_tracer = tracer
        return tracer

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.sampling import (
            ALWAYS_OFF,
            ALWAYS_ON,
            ParentBasedTraceIdRatio,
            TraceIdRatioBased,
        )

        # --- sampler ---
        if config.sampler == "always_off":
            sampler = ALWAYS_OFF
        elif config.sampler == "always_on":
            sampler = ALWAYS_ON
        elif config.sampler == "parentbased_traceidratio":
            sampler = ParentBasedTraceIdRatio(config.sample_rate)
        else:
            # Legacy path: honor sampling_strategy for backward compat
            ratio_sampler = TraceIdRatioBased(config.sample_rate)
            if config.sampling_strategy == "parentbased":
                sampler = ParentBasedTraceIdRatio(config.sample_rate)
            else:
                sampler = ratio_sampler

        # --- resource ---
        resource = Resource.create({
            "service.name": config.service_name,
            "service.version": config.service_version,
            "service.namespace": "ocr-local",
            "deployment.environment": config.environment,
        })
        provider = TracerProvider(resource=resource, sampler=sampler)

        # --- exporter ---
        if config.exporter in ("otlp", "jaeger"):
            if config.exporter == "jaeger":
                logger.warning(
                    "OTel exporter 'jaeger' is deprecated; use 'otlp' with "
                    "OTEL_EXPORTER_OTLP_ENDPOINT pointing at your Jaeger OTLP "
                    "port (4317). Falling back to 'otlp'."
                )

            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            exporter_kwargs: Dict[str, Any] = {"endpoint": config.endpoint}

            # Pass headers for auth (e.g. Grafana Cloud, Datadog)
            if config.headers:
                exporter_kwargs["headers"] = tuple(config.headers.items())

            # Jaeger legacy path: default to insecure if no explicit TLS
            if config.exporter == "jaeger":
                exporter_kwargs.setdefault("insecure", True)

            exporter_obj = OTLPSpanExporter(**exporter_kwargs)
            provider.add_span_processor(BatchSpanProcessor(exporter_obj))
        else:
            # console exporter (default for development)
            from opentelemetry.sdk.trace.export import (
                ConsoleSpanExporter,
                SimpleSpanProcessor,
            )

            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

        trace.set_tracer_provider(provider)
        tracer = trace.get_tracer(config.service_name, config.service_version)
        with _tracer_lock:
            _global_tracer = tracer
        logger.info(
            "OpenTelemetry tracing initialised "
            "(exporter=%s, sampler=%s, rate=%.2f, endpoint=%s, env=%s)",
            config.exporter,
            config.sampler,
            config.sample_rate,
            config.endpoint,
            config.environment,
        )
        return tracer

    except ImportError:
        logger.info(
            "OpenTelemetry SDK not available -- using LocalTracer fallback. "
            "Install opentelemetry-api, opentelemetry-sdk, and "
            "opentelemetry-exporter-otlp-proto-grpc for production tracing."
        )
        tracer = LocalTracer(service_name=config.service_name)
        with _tracer_lock:
            _global_tracer = tracer
        return tracer


def get_tracer() -> Any:
    """Return the global tracer, creating a default :class:`LocalTracer` if needed."""
    global _global_tracer
    with _tracer_lock:
        if _global_tracer is None:
            _global_tracer = LocalTracer()
        return _global_tracer


# ---------------------------------------------------------------------------
# FastAPI integration
# ---------------------------------------------------------------------------


def configure_tracing(app: Any, config: Optional[TracingConfig] = None) -> Any:
    """Initialise tracing and optionally instrument a FastAPI application.

    This is the recommended single entry point for ``api/main.py``::

        from api.tracing import configure_tracing
        configure_tracing(app)

    If ``opentelemetry-instrumentation-fastapi`` is installed, request/
    response spans are created automatically.  Otherwise tracing is still
    active for manual ``trace_operation`` usage.

    Returns the active tracer instance.
    """
    tracer = init_tracing(config)

    # Attempt FastAPI auto-instrumentation (optional dependency)
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
        logger.info("FastAPI auto-instrumentation enabled")
    except ImportError:
        logger.debug(
            "opentelemetry-instrumentation-fastapi not installed; "
            "skipping auto-instrumentation"
        )
    except Exception as exc:
        logger.warning("FastAPI auto-instrumentation failed: %s", exc)

    return tracer


# ---------------------------------------------------------------------------
# Convenience context manager / decorator
# ---------------------------------------------------------------------------


@contextmanager
def trace_operation(
    name: str,
    *,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Optional[Dict[str, Any]] = None,
):
    """Context manager that creates a span for the enclosed block.

    Works transparently with both the OTel SDK tracer and :class:`LocalTracer`.

    Example::

        with trace_operation("ocr_page", attributes={"page": 3}):
            result = run_ocr(page)
    """
    tracer = get_tracer()

    if isinstance(tracer, LocalTracer):
        span = tracer.start_span(name, kind=kind, attributes=attributes)
        try:
            yield span
            if span.status == SpanStatus.UNSET:
                span.set_status(SpanStatus.OK)
        except Exception:
            span.set_status(SpanStatus.ERROR, "exception")
            raise
        finally:
            span.end()
    else:
        # OTel SDK tracer path
        try:
            from opentelemetry.trace import SpanKind as OtelSpanKind
            from opentelemetry.trace import StatusCode

            _kind_map = {
                SpanKind.CLIENT: OtelSpanKind.CLIENT,
                SpanKind.SERVER: OtelSpanKind.SERVER,
                SpanKind.PRODUCER: OtelSpanKind.PRODUCER,
                SpanKind.CONSUMER: OtelSpanKind.CONSUMER,
                SpanKind.INTERNAL: OtelSpanKind.INTERNAL,
            }

            otel_kind = _kind_map.get(kind, OtelSpanKind.INTERNAL)
            with tracer.start_as_current_span(name, kind=otel_kind) as span:
                if attributes:
                    for k, v in attributes.items():
                        span.set_attribute(k, v)
                try:
                    yield span
                    span.set_status(StatusCode.OK)
                except Exception:
                    span.set_status(StatusCode.ERROR, "exception")
                    raise
        except ImportError:
            # Shouldn't happen but graceful fallback
            yield None
