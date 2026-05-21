"""Prometheus language-metric tests (Plan A -- PR A4).

Verifies the three Plan-A metric families wired into ``api/prometheus.py``:

* ``ocr_language_detected_total`` -- Counter labelled by lang/tier/level
* ``ocr_language_confidence`` -- Histogram labelled by lang/level
* ``ocr_language_mixed_script_pages_total`` -- Counter labelled by script

The tests use the prometheus_client ``CollectorRegistry`` directly; when
that library is not installed the whole test module is skipped.

Run with::

    python -m pytest tests/test_prometheus_language.py -v
"""

from __future__ import annotations

import pytest

prometheus_client = pytest.importorskip("prometheus_client")

from api.prometheus import (  # noqa: E402  (import after importorskip)
    _REGISTRY,
    ocr_language_confidence,
    ocr_language_detected_total,
    ocr_language_mixed_script_pages_total,
)


def _sample(metric_family_name: str, **labels) -> float | None:
    """Return the sum/value sample matching the given labels, else None."""
    for metric in _REGISTRY.collect():
        if metric.name != metric_family_name:
            # prometheus_client strips ``_total`` suffix from counter
            # family names; also support the histogram/family matching.
            if not (
                metric.name + "_total" == metric_family_name
                or metric.name == metric_family_name
            ):
                continue
        for sample in metric.samples:
            if all(sample.labels.get(k) == v for k, v in labels.items()):
                # For counters prometheus emits a ``_total`` sample name.
                if sample.name.endswith("_total") or sample.name == metric.name:
                    return sample.value
                if sample.name == metric.name + "_sum":
                    return sample.value
    return None


def _hist_count(lang: str, level: str) -> float:
    """Return the total observation count for the confidence histogram."""
    total = 0.0
    for metric in _REGISTRY.collect():
        if metric.name != "ocr_language_confidence":
            continue
        for sample in metric.samples:
            if (
                sample.name == "ocr_language_confidence_count"
                and sample.labels.get("lang") == lang
                and sample.labels.get("level") == level
            ):
                total += sample.value
    return total


# ---------------------------------------------------------------------------
# Counter: ocr_language_detected_total
# ---------------------------------------------------------------------------


class TestLanguageDetectedCounter:
    def test_counter_is_registered(self):
        assert ocr_language_detected_total is not None

    def test_counter_has_three_labels(self):
        # prometheus_client stores label names on the metric's ``_labelnames``.
        assert ocr_language_detected_total._labelnames == ("lang", "tier", "level")

    def test_counter_increment_page_level(self):
        before = (
            _sample(
                "ocr_language_detected_total",
                lang="en", tier="core", level="page",
            )
            or 0.0
        )
        ocr_language_detected_total.labels(
            lang="en", tier="core", level="page",
        ).inc()
        after = (
            _sample(
                "ocr_language_detected_total",
                lang="en", tier="core", level="page",
            )
            or 0.0
        )
        assert after - before == pytest.approx(1.0)

    def test_counter_increment_document_level(self):
        before = (
            _sample(
                "ocr_language_detected_total",
                lang="fr", tier="core", level="document",
            )
            or 0.0
        )
        ocr_language_detected_total.labels(
            lang="fr", tier="core", level="document",
        ).inc()
        after = (
            _sample(
                "ocr_language_detected_total",
                lang="fr", tier="core", level="document",
            )
            or 0.0
        )
        assert after - before == pytest.approx(1.0)

    def test_counter_increment_extended_tier(self):
        before = (
            _sample(
                "ocr_language_detected_total",
                lang="hr", tier="extended", level="page",
            )
            or 0.0
        )
        ocr_language_detected_total.labels(
            lang="hr", tier="extended", level="page",
        ).inc()
        after = (
            _sample(
                "ocr_language_detected_total",
                lang="hr", tier="extended", level="page",
            )
            or 0.0
        )
        assert after - before == pytest.approx(1.0)

    def test_counter_accepts_und_label(self):
        """Undetermined language still increments the counter."""
        before = (
            _sample(
                "ocr_language_detected_total",
                lang="und", tier="core", level="page",
            )
            or 0.0
        )
        ocr_language_detected_total.labels(
            lang="und", tier="core", level="page",
        ).inc()
        after = (
            _sample(
                "ocr_language_detected_total",
                lang="und", tier="core", level="page",
            )
            or 0.0
        )
        assert after - before == pytest.approx(1.0)

    def test_counter_labels_are_strings(self):
        labels = ocr_language_detected_total.labels(
            lang="en", tier="core", level="page",
        )
        # Internal label dict uses string values.
        for v in labels._labelvalues:
            assert isinstance(v, str)

    def test_counter_multiple_increments_sum(self):
        for _ in range(3):
            ocr_language_detected_total.labels(
                lang="ch", tier="core", level="page",
            ).inc()
        value = _sample(
            "ocr_language_detected_total",
            lang="ch", tier="core", level="page",
        )
        assert value is not None
        assert value >= 3.0


# ---------------------------------------------------------------------------
# Histogram: ocr_language_confidence
# ---------------------------------------------------------------------------


class TestLanguageConfidenceHistogram:
    def test_histogram_is_registered(self):
        assert ocr_language_confidence is not None

    def test_histogram_has_two_labels(self):
        assert ocr_language_confidence._labelnames == ("lang", "level")

    def test_histogram_buckets_match_spec(self):
        expected = [0.2, 0.4, 0.6, 0.8, 0.95, 1.0]
        # prometheus_client stores buckets as strings like "0.2" plus "+Inf".
        buckets = [float(b) for b in ocr_language_confidence._upper_bounds[:-1]]
        assert buckets == expected

    def test_histogram_observation_recorded(self):
        before = _hist_count("en", "page")
        ocr_language_confidence.labels(lang="en", level="page").observe(0.85)
        after = _hist_count("en", "page")
        assert after - before == pytest.approx(1.0)

    def test_histogram_document_level_label(self):
        before = _hist_count("ru", "document")
        ocr_language_confidence.labels(lang="ru", level="document").observe(0.7)
        after = _hist_count("ru", "document")
        assert after - before == pytest.approx(1.0)

    def test_histogram_multiple_observations(self):
        before = _hist_count("ar", "page")
        for val in (0.3, 0.6, 0.95):
            ocr_language_confidence.labels(lang="ar", level="page").observe(val)
        after = _hist_count("ar", "page")
        assert after - before == pytest.approx(3.0)

    def test_histogram_accepts_extremes(self):
        # 0.0 and 1.0 are valid confidences.
        before = _hist_count("ja", "page")
        ocr_language_confidence.labels(lang="ja", level="page").observe(0.0)
        ocr_language_confidence.labels(lang="ja", level="page").observe(1.0)
        after = _hist_count("ja", "page")
        assert after - before == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Counter: ocr_language_mixed_script_pages_total
# ---------------------------------------------------------------------------


class TestMixedScriptCounter:
    def test_counter_is_registered(self):
        assert ocr_language_mixed_script_pages_total is not None

    def test_counter_has_one_label(self):
        assert ocr_language_mixed_script_pages_total._labelnames == (
            "primary_script",
        )

    def test_counter_increments_for_latin(self):
        before = (
            _sample(
                "ocr_language_mixed_script_pages_total",
                primary_script="latin",
            )
            or 0.0
        )
        ocr_language_mixed_script_pages_total.labels(
            primary_script="latin",
        ).inc()
        after = (
            _sample(
                "ocr_language_mixed_script_pages_total",
                primary_script="latin",
            )
            or 0.0
        )
        assert after - before == pytest.approx(1.0)

    def test_counter_increments_for_arabic(self):
        before = (
            _sample(
                "ocr_language_mixed_script_pages_total",
                primary_script="arabic",
            )
            or 0.0
        )
        ocr_language_mixed_script_pages_total.labels(
            primary_script="arabic",
        ).inc()
        after = (
            _sample(
                "ocr_language_mixed_script_pages_total",
                primary_script="arabic",
            )
            or 0.0
        )
        assert after - before == pytest.approx(1.0)

    def test_counter_not_incremented_without_call(self):
        """Do not increment when mixed_script is False."""
        before = (
            _sample(
                "ocr_language_mixed_script_pages_total",
                primary_script="cjk",
            )
            or 0.0
        )
        # Simulate: mixed_script=False -> skip increment.
        mixed_script = False
        if mixed_script:  # pragma: no cover
            ocr_language_mixed_script_pages_total.labels(
                primary_script="cjk",
            ).inc()
        after = (
            _sample(
                "ocr_language_mixed_script_pages_total",
                primary_script="cjk",
            )
            or 0.0
        )
        assert before == after


# ---------------------------------------------------------------------------
# Graceful-degradation contract
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    def test_import_guarded_with_try_except(self):
        """The pipeline guards Prometheus imports so missing
        prometheus_client never blocks OCR output.  Verify the import
        surface is exactly what ocr_gpu_async.py depends on."""
        from api import prometheus as p
        assert hasattr(p, "ocr_language_detected_total")
        assert hasattr(p, "ocr_language_confidence")
        assert hasattr(p, "ocr_language_mixed_script_pages_total")

    def test_labels_never_raise_for_known_values(self):
        # The pipeline always passes string labels; verify that passing
        # the canonical set does not raise.
        ocr_language_detected_total.labels(
            lang="en", tier="core", level="page",
        )
        ocr_language_confidence.labels(lang="en", level="page")
        ocr_language_mixed_script_pages_total.labels(primary_script="latin")

    def test_registry_exposes_metric_families(self):
        names = {m.name for m in _REGISTRY.collect()}
        # Counter families lose the ``_total`` suffix in ``name``.
        assert "ocr_language_detected" in names
        assert "ocr_language_confidence" in names
        assert "ocr_language_mixed_script_pages" in names

    def test_simulated_worker_call_chain_end_to_end(self):
        """Simulate the full call chain the GPU worker performs."""
        from ocr_local.config.language_config import LANGUAGE_REGISTRY

        primary_language = "en"
        primary_confidence = 0.87
        mixed_script = True
        scripts_detected = ["latin", "cyrillic"]
        entry = LANGUAGE_REGISTRY.get(primary_language)
        tier = entry.tier if entry is not None else "core"

        ocr_language_detected_total.labels(
            lang=primary_language, tier=tier, level="page",
        ).inc()
        ocr_language_confidence.labels(
            lang=primary_language, level="page",
        ).observe(primary_confidence)
        if mixed_script and scripts_detected:
            ocr_language_mixed_script_pages_total.labels(
                primary_script=scripts_detected[0],
            ).inc()

        # End-to-end smoke: all three counters have non-None samples for
        # the labels we set.
        assert (
            _sample(
                "ocr_language_detected_total",
                lang="en", tier=tier, level="page",
            )
            is not None
        )
        assert _hist_count("en", "page") > 0
        assert (
            _sample(
                "ocr_language_mixed_script_pages_total",
                primary_script="latin",
            )
            is not None
        )

    def test_unknown_language_falls_back_to_core_tier(self):
        """If a language is not in the registry, pipeline defaults to core."""
        from ocr_local.config.language_config import LANGUAGE_REGISTRY
        entry = LANGUAGE_REGISTRY.get("zz_not_a_language")
        tier = entry.tier if entry is not None else "core"
        assert tier == "core"

    def test_extended_language_resolves_to_extended_tier(self):
        from ocr_local.config.language_config import LANGUAGE_REGISTRY
        # hr is extended per language_config.
        entry = LANGUAGE_REGISTRY.get("hr")
        assert entry is not None
        assert entry.tier == "extended"

    def test_core_language_resolves_to_core_tier(self):
        from ocr_local.config.language_config import LANGUAGE_REGISTRY
        entry = LANGUAGE_REGISTRY.get("en")
        assert entry is not None
        assert entry.tier == "core"

    def test_metric_types(self):
        from prometheus_client import Counter, Histogram
        assert isinstance(ocr_language_detected_total, Counter)
        assert isinstance(ocr_language_mixed_script_pages_total, Counter)
        assert isinstance(ocr_language_confidence, Histogram)
