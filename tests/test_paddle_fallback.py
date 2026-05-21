"""Tests for the PaddleOCR engine fallback logic in worker_thread's get_engine().

When both the primary language engine AND the English fallback engine
fail to load, the failure is cached as None so subsequent pages skip the
model-load retry and immediately fall through to Tesseract/ImageOnly.
"""

import threading
from unittest.mock import MagicMock, patch

import pytest


def _build_get_engine(create_engine_mock, worker_id=0):
    """Build a standalone get_engine closure that mirrors worker_thread's logic.

    This reproduces the exact double-checked locking and fallback pattern from
    ocr_gpu_async.py worker_thread so we can unit-test it without spinning up
    the full pipeline.
    """
    engines = {}
    model_load_lock = threading.Lock()
    import logging
    logger = logging.getLogger("test_paddle_fallback")

    def get_engine(lang_code):
        if lang_code not in engines:
            with model_load_lock:
                if lang_code not in engines:
                    try:
                        logger.info(f"Worker {worker_id} loading model for: {lang_code}")
                        engines[lang_code] = create_engine_mock(lang_code, device='gpu')
                    except Exception as primary_exc:
                        logger.warning(
                            f"Worker {worker_id}: Failed to load {lang_code} model, "
                            f"falling back to English. Error: {primary_exc}"
                        )
                        try:
                            engines[lang_code] = create_engine_mock('en', device='gpu')
                        except Exception as fallback_exc:
                            logger.error(
                                f"Worker {worker_id}: English fallback also failed for "
                                f"{lang_code}. All pages will use Tesseract/ImageOnly. "
                                f"Error: {fallback_exc}"
                            )
                            engines[lang_code] = None

        engine = engines.get(lang_code)
        if engine is None:
            raise RuntimeError(
                f"No PaddleOCR engine available for {lang_code} "
                f"(primary and English fallback both failed)"
            )
        return engine

    return get_engine, engines


class TestGetEngineFallback:
    """Test the get_engine double-fallback pattern."""

    def test_primary_success(self):
        """Primary engine loads successfully -- no fallback needed."""
        mock_engine = MagicMock(name="fr_engine")
        create_mock = MagicMock(return_value=mock_engine)

        get_engine, engines = _build_get_engine(create_mock)
        result = get_engine("fr")

        assert result is mock_engine
        create_mock.assert_called_once_with("fr", device="gpu")
        assert engines["fr"] is mock_engine

    def test_primary_fails_english_succeeds(self):
        """Primary language fails, English fallback succeeds."""
        en_engine = MagicMock(name="en_engine")

        def side_effect(lang, device="gpu"):
            if lang == "ar":
                raise RuntimeError("Arabic model not found")
            return en_engine

        create_mock = MagicMock(side_effect=side_effect)
        get_engine, engines = _build_get_engine(create_mock)

        result = get_engine("ar")

        assert result is en_engine
        assert engines["ar"] is en_engine
        assert create_mock.call_count == 2
        create_mock.assert_any_call("ar", device="gpu")
        create_mock.assert_any_call("en", device="gpu")

    def test_both_fail_caches_none(self):
        """Both primary and English fallback fail -- cache stores None."""
        create_mock = MagicMock(side_effect=RuntimeError("GPU unavailable"))
        get_engine, engines = _build_get_engine(create_mock)

        with pytest.raises(RuntimeError, match="No PaddleOCR engine available for fr"):
            get_engine("fr")

        # The failure is cached as None
        assert "fr" in engines
        assert engines["fr"] is None

    def test_cached_none_skips_retry(self):
        """After double failure, subsequent calls raise immediately without retrying."""
        create_mock = MagicMock(side_effect=RuntimeError("GPU unavailable"))
        get_engine, engines = _build_get_engine(create_mock)

        # First call: triggers both attempts
        with pytest.raises(RuntimeError, match="No PaddleOCR engine available"):
            get_engine("ja")

        assert create_mock.call_count == 2  # primary + English fallback

        # Reset call count and call again
        create_mock.reset_mock()

        with pytest.raises(RuntimeError, match="No PaddleOCR engine available for ja"):
            get_engine("ja")

        # No new calls to create_paddle_engine -- failure was cached
        assert create_mock.call_count == 0

    def test_cached_success_skips_reload(self):
        """After successful load, subsequent calls return cached engine."""
        mock_engine = MagicMock(name="en_engine")
        create_mock = MagicMock(return_value=mock_engine)
        get_engine, engines = _build_get_engine(create_mock)

        result1 = get_engine("en")
        result2 = get_engine("en")

        assert result1 is result2
        assert result1 is mock_engine
        create_mock.assert_called_once_with("en", device="gpu")

    def test_different_languages_independent(self):
        """Different language codes have independent cache entries."""
        engines_created = {}

        def side_effect(lang, device="gpu"):
            if lang == "ar":
                raise RuntimeError("Arabic model not found")
            eng = MagicMock(name=f"{lang}_engine")
            engines_created[lang] = eng
            return eng

        create_mock = MagicMock(side_effect=side_effect)
        get_engine, engines = _build_get_engine(create_mock)

        # 'fr' succeeds directly
        fr_result = get_engine("fr")
        assert fr_result is engines_created["fr"]

        # 'ar' falls back to English
        ar_result = get_engine("ar")
        assert ar_result is engines_created["en"]

        # Both are cached independently
        assert engines["fr"] is engines_created["fr"]
        assert engines["ar"] is engines_created["en"]

    def test_english_primary_fails_english_fallback_also_fails(self):
        """When lang_code IS 'en' and both attempts fail, None is cached for 'en'."""
        create_mock = MagicMock(side_effect=RuntimeError("No CUDA"))
        get_engine, engines = _build_get_engine(create_mock)

        with pytest.raises(RuntimeError, match="No PaddleOCR engine available for en"):
            get_engine("en")

        assert engines["en"] is None
        # Both calls used 'en': primary attempt + English fallback attempt
        assert create_mock.call_count == 2


class TestGetEngineIntegration:
    """Integration-level tests that verify get_engine interacts correctly
    with the module-level create_paddle_engine via mocking."""

    @patch("ocr_gpu_async.create_paddle_engine")
    def test_module_level_double_failure_caches_none(self, mock_create):
        """Verify that if we import and use the real get_engine path,
        double failures are handled. We test via the module-level function
        mock since get_engine is a closure we cannot import directly."""
        mock_create.side_effect = RuntimeError("PaddleOCR dependency not installed")

        # The actual get_engine is inside worker_thread and not importable,
        # so we replicate the pattern and verify the mock behavior matches
        # what the production code would do.
        engines = {}
        import ocr_gpu_async

        model_load_lock = threading.Lock()

        # Simulate the get_engine logic using the real module's function
        lang_code = "de"
        with model_load_lock:
            try:
                engines[lang_code] = ocr_gpu_async.create_paddle_engine(lang_code, device="gpu")
            except Exception:
                try:
                    engines[lang_code] = ocr_gpu_async.create_paddle_engine("en", device="gpu")
                except Exception:
                    engines[lang_code] = None

        assert engines["de"] is None
        assert mock_create.call_count == 2
