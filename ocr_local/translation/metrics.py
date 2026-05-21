"""Prometheus metrics for the translation pipeline -- Plan B Wave M1.

Mirrors the optional-import pattern used in
``ocr_local.infra.ocr_metrics``: when ``prometheus_client`` is missing
(common on the SDK CI lane), the metric handles fall back to safe
no-op shims so callers never need to guard imports.

Metrics:
    ``ocr_translation_chars_total``  -- Counter labelled by tenant/engine
    ``ocr_translation_tokens_total`` -- Counter labelled by tenant/engine
    ``ocr_translation_duration_seconds`` -- Histogram labelled by engine
"""

from __future__ import annotations

__all__ = [
    "ocr_translation_chars_total",
    "ocr_translation_tokens_total",
    "ocr_translation_duration_seconds",
    "record_translation_chars",
    "record_translation_tokens",
    "record_translation_duration",
]

try:
    from prometheus_client import Counter, Histogram

    _HAVE_PROMETHEUS = True
except ImportError:  # pragma: no cover - exercised by reload test
    _HAVE_PROMETHEUS = False

    class _Noop:
        """Safe no-op stand-in for Prometheus Counter/Histogram."""

        def labels(self, **_: object) -> "_Noop":
            return self

        def inc(self, n: float = 1) -> None:
            return None

        def observe(self, n: float) -> None:
            return None

    def Counter(*_args: object, **_kwargs: object) -> _Noop:  # type: ignore[no-redef]
        return _Noop()

    def Histogram(*_args: object, **_kwargs: object) -> _Noop:  # type: ignore[no-redef]
        return _Noop()


ocr_translation_chars_total = Counter(
    "ocr_translation_chars_total",
    "Characters processed by translation engine",
    ["tenant", "engine"],
)

ocr_translation_tokens_total = Counter(
    "ocr_translation_tokens_total",
    "Tokens processed by translation engine",
    ["tenant", "engine"],
)

ocr_translation_duration_seconds = Histogram(
    "ocr_translation_duration_seconds",
    "Translation engine processing time in seconds",
    ["engine"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)


def record_translation_chars(tenant: str, engine: str, n: int) -> None:
    """Record ``n`` characters processed by ``engine`` for ``tenant``."""
    ocr_translation_chars_total.labels(tenant=tenant, engine=engine).inc(n)


def record_translation_tokens(tenant: str, engine: str, n: int) -> None:
    """Record ``n`` tokens processed by ``engine`` for ``tenant``."""
    ocr_translation_tokens_total.labels(tenant=tenant, engine=engine).inc(n)


def record_translation_duration(engine: str, seconds: float) -> None:
    """Record an engine processing duration sample (in seconds)."""
    ocr_translation_duration_seconds.labels(engine=engine).observe(seconds)
