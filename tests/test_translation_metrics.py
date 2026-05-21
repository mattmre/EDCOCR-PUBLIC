"""Tests for ocr_local.translation.metrics -- Plan B Wave M1 PR2."""

from __future__ import annotations

import builtins
import importlib
import sys


def test_record_chars_no_error():
    from ocr_local.translation.metrics import record_translation_chars

    # Must not raise -- no-op if prometheus_client is missing.
    record_translation_chars("t1", "passthrough", 100)


def test_record_tokens_no_error():
    from ocr_local.translation.metrics import record_translation_tokens

    record_translation_tokens("t1", "passthrough", 25)


def test_record_duration_no_error():
    from ocr_local.translation.metrics import record_translation_duration

    record_translation_duration("passthrough", 1.5)


def test_noop_when_prometheus_missing(monkeypatch):
    """When prometheus_client is unimportable, the metrics module still loads."""
    import ocr_local.translation.metrics as original_metrics

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "prometheus_client" or name.startswith("prometheus_client."):
            raise ImportError("simulated absence of prometheus_client")
        return real_import(name, globals, locals, fromlist, level)

    # Save existing prometheus_client modules so we can restore them.
    saved_modules = {
        k: v
        for k, v in sys.modules.items()
        if k == "prometheus_client" or k.startswith("prometheus_client.")
    }
    saved_metrics = sys.modules.get("ocr_local.translation.metrics")

    try:
        # Drop cached modules so the reimport triggers the ImportError path.
        for k in list(sys.modules):
            if k == "prometheus_client" or k.startswith("prometheus_client."):
                del sys.modules[k]
        sys.modules.pop("ocr_local.translation.metrics", None)
        monkeypatch.setattr(builtins, "__import__", fake_import)

        metrics = importlib.import_module("ocr_local.translation.metrics")
        # All record_* helpers must succeed under the no-op fallback.
        metrics.record_translation_chars("t1", "passthrough", 10)
        metrics.record_translation_tokens("t1", "passthrough", 3)
        metrics.record_translation_duration("passthrough", 0.5)
        assert metrics._HAVE_PROMETHEUS is False
    finally:
        # Restore real prometheus_client modules and the original metrics
        # module instance so the existing Counter/Histogram registrations
        # in the global Prometheus registry are not duplicated.
        monkeypatch.setattr(builtins, "__import__", real_import)
        sys.modules.update(saved_modules)
        if saved_metrics is not None:
            sys.modules["ocr_local.translation.metrics"] = saved_metrics
        else:
            sys.modules.pop("ocr_local.translation.metrics", None)
        # Sanity: original module still importable and unaffected.
        assert original_metrics.ocr_translation_chars_total is not None


def test_counters_exported():
    import ocr_local.translation.metrics as m

    assert hasattr(m, "ocr_translation_chars_total")
    assert hasattr(m, "ocr_translation_tokens_total")


def test_histogram_exported():
    import ocr_local.translation.metrics as m

    assert hasattr(m, "ocr_translation_duration_seconds")
